from collections import OrderedDict
from typing import List, Dict

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn, Tensor

from kge import Configurable, Config, Dataset
from kge.model.ensemble.aggregation_data import create_aggregation_dataloader
from kge.model.ensemble.model_manager import ModelManager
from kge.model.ensemble.util import EmbeddingType, EmbeddingTarget


def concat_embeds(embeds):
    """
    Takes embeddings of the form n times k_i, where n is the number of samples and k_i is the
    model specific embedding size.
    Return the concatenation of embeddings with form n times k, where k is the sum of k_i for 1 <= i <= m.
    :param embeds:
    :return:
    """
    embeds_list = []
    for model_idx in embeds.keys():
        embeds_list.append(embeds[model_idx])
    out = torch.cat(embeds_list, 1)
    return out


def pad_embeds(embeds, padding_length):
    """
    Pads all embeddings with zero to embedding length padding_length.
    :param embeds:
    :param padding_length:
    :return:
    """
    padded_embeds = {}
    for model_idx in embeds:
        model_embeds = embeds[model_idx]
        last_dim = model_embeds.size()[1]
        padded_embed = F.pad(input=model_embeds, pad=(0, padding_length - last_dim), mode='constant', value=0)
        padded_embeds[model_idx] = padded_embed
    return padded_embeds


def avg_embeds(embeds):
    """
    Takes embeddings of the same form n times k and return the average mean over all embeddings.
    :param embeds:
    :return:
    """
    embed_list = []
    for model_idx in embeds:
        embed = embeds[model_idx]
        n = embed.size()[0]
        embed = embed.view(n, 1, -1)
        embed_list.append(embed)
    out = torch.cat(embed_list, 1)
    out = torch.mean(out, 1)
    return out


class AggregationBase(nn.Module, Configurable):

    def __init__(self, model_manager: ModelManager, config: Config, configuration_key, parent_configuration_key):
        """
        Initializes basic dim reduction variables.
        Updates the reduced dimensionality in embedding ensemble if entity_reduction and relation_reduction
        are given for the reduction model, else do a concatenation
        If parameters are already set, do not update them
        :param config:
        :param configuration_key:
        :param parent_configuration_key:
        """
        Configurable.__init__(self, config, configuration_key)
        nn.Module.__init__(self)
        self.model_manager = model_manager

        # compute aggregated entity and relation dimensions
        self.entity_agg_dim = 0
        self.relation_agg_dim = 0
        self.compute_dims()

        # write agg dims for evaluator in embedding ensemble config
        self.config.set(parent_configuration_key + ".entity_agg_dim", self.entity_agg_dim)
        self.config.set(parent_configuration_key + ".relation_agg_dim", self.relation_agg_dim)

    def compute_dims(self):
        # compute concatenated agg dim
        embed_dims = self.model_manager.get_model_dims()
        for model_idx in range(self.model_manager.num_models()):
            self.entity_agg_dim += embed_dims[model_idx][EmbeddingType.Entity]
            self.relation_agg_dim += embed_dims[model_idx][EmbeddingType.Relation]

        # get entity and relation reduction by percentage
        entity_reduction = 1.0
        relation_reduction = 1.0
        if self.has_option("entity_reduction"):
            entity_reduction = self.get_option("entity_reduction")
        if self.has_option("relation_reduction"):
            relation_reduction = self.get_option("relation_reduction")

        self.entity_agg_dim = round(self.entity_agg_dim * entity_reduction)
        self.relation_agg_dim = round(self.relation_agg_dim * relation_reduction)

    def aggregate(self, target: EmbeddingTarget, indexes: Tensor = None):
        """
        Applies an aggregation function for the given indexes and the specified target of
        embedder
        :param target: Can have the values "s", "p" and "o"
        :param indexes: Tensor of entity or relation indexes
        :return: Aggregated embeddings of multiple models
        """
        raise NotImplementedError

    def train_aggregation(self):
        raise NotImplementedError

    def penalty(self, **kwargs) -> List[Tensor]:
        return []


class Concatenation(AggregationBase):

    def __init__(self, model_manager: ModelManager, config, parent_configuration_key):
        AggregationBase.__init__(self, model_manager, config, "concat", parent_configuration_key)

    def aggregate(self, target: EmbeddingTarget, indexes: Tensor = None):
        t = self.model_manager.fetch_model_embeddings(target, indexes)
        res = concat_embeds(t)
        return res

    def train_aggregation(self):
        pass


class MeanReduction(AggregationBase):

    def __init__(self, model_manager: ModelManager, config, parent_configuration_key):
        AggregationBase.__init__(self, model_manager, config, "mean", parent_configuration_key)

    def compute_dims(self):
        # find the longest entity and relation embeddings
        embed_dims = self.model_manager.get_model_dims()
        for model_idx in range(self.model_manager.num_models()):
            self.entity_agg_dim = max(self.entity_agg_dim, embed_dims[model_idx][EmbeddingType.Entity])
            self.relation_agg_dim = max(self.relation_agg_dim, embed_dims[model_idx][EmbeddingType.Relation])

    def aggregate(self, target: EmbeddingTarget, indexes: Tensor = None):
        t = self.model_manager.fetch_model_embeddings(target, indexes)
        if target == EmbeddingTarget.Subject or target == EmbeddingTarget.Object:
            t = pad_embeds(t, self.entity_agg_dim)
        elif target == EmbeddingTarget.Predicate:
            t = pad_embeds(t, self.relation_agg_dim)
        else:
            raise ValueError("Unknown target embedding: " + str(target))
        t = avg_embeds(t)
        return t

    def train_aggregation(self):
        pass


class PcaReduction(AggregationBase):

    def __init__(self, model_manager: ModelManager, dataset, config, parent_configuration_key):
        AggregationBase.__init__(self, model_manager, config, "pca", parent_configuration_key)

        self._entity_embedder = torch.nn.Embedding(dataset.num_entities(), self.entity_agg_dim)
        self._relation_embedder = torch.nn.Embedding(dataset.num_relations(), self.relation_agg_dim)

    def aggregate(self, target: EmbeddingTarget, indexes: Tensor = None):
        if target == EmbeddingTarget.Subject or target == EmbeddingTarget.Object:
            if indexes is None:
                entity_indexes = torch.arange(
                    self.dataset.num_entities(), dtype=torch.long, device=self._entity_embedder.weight.device
                )
                return self._entity_embedder(entity_indexes)
            else:
                return self._entity_embedder(indexes.long())
        elif target == EmbeddingTarget.Predicate:
            if indexes is None:
                relation_indexes = torch.arange(
                    self.dataset.num_relations(), dtype=torch.long, device=self._entity_embedder.weight.device
                )
                return self._relation_embedder(relation_indexes)
            else:
                return self._relation_embedder(indexes.long())
        else:
            raise ValueError("Unknown target embedding " + str(target))

    def train_aggregation(self):
        # create pca models
        entity_pca = PCA(n_components=self.entity_agg_dim)
        relation_pca = PCA(n_components=self.relation_agg_dim)

        # fetch and preprocess embeddings
        entity_embeds = self.model_manager.fetch_model_embeddings(EmbeddingTarget.Subject)
        relation_embeds = self.model_manager.fetch_model_embeddings(EmbeddingTarget.Predicate)
        entity_embeds = concat_embeds(entity_embeds)
        relation_embeds = concat_embeds(relation_embeds)

        # prior standardization
        entity_embeds = StandardScaler().fit_transform(entity_embeds)
        relation_embeds = StandardScaler().fit_transform(relation_embeds)

        # apply pca
        entity_embeds = entity_pca.fit_transform(entity_embeds)
        entity_embeds = torch.tensor(entity_embeds, dtype=torch.float, device=self.config.get("job.device"))
        relation_embeds = relation_pca.fit_transform(relation_embeds)
        relation_embeds = torch.tensor(relation_embeds, dtype=torch.float, device=self.config.get("job.device"))

        # store embeddings
        self._entity_embedder = nn.Embedding.from_pretrained(entity_embeds)
        self._relation_embedder = nn.Embedding.from_pretrained(relation_embeds)


class AutoencoderReduction(AggregationBase):

    def __init__(self, model_manager: ModelManager, config: Config, parent_configuration_key):
        AggregationBase.__init__(self, model_manager, config, "autoencoder_reduction", parent_configuration_key)

        # get training parameters
        self.epochs = self.get_option("epochs")
        self.lr = self.get_option("lr")
        self.batch_size = self.get_option("batch_size")
        self.weight_decay = self.get_option("weight_decay")

        # create autoencoders
        entity_autoencoders = []
        relation_autoencoders = []
        embed_dims = self.model_manager.get_model_dims()
        for model_idx in range(self.model_manager.num_models()):
            entity_dim_in = embed_dims[model_idx][EmbeddingType.Entity]
            relation_dim_in = embed_dims[model_idx][EmbeddingType.Relation]
            entity_autoencoders.append(Autoencoder(config, entity_dim_in, self.entity_agg_dim))
            relation_autoencoders.append(Autoencoder(config, relation_dim_in, self.relation_agg_dim))
        self.entity_models = nn.ModuleList(entity_autoencoders)
        self.relation_models = nn.ModuleList(relation_autoencoders)

    def aggregate(self, target, indexes: Tensor = None):
        embeds = self.model_manager.fetch_model_embeddings(target, indexes)
        if target == EmbeddingTarget.Subject or target == EmbeddingTarget.Object:
            res = self.encode(EmbeddingType.Entity, embeds)
        elif target == EmbeddingTarget.Predicate:
            res = self.encode(EmbeddingType.Relation, embeds)
        else:
            raise ValueError("Unknown target embedding:" + str(target))
        return res.detach()

    def encode(self, e_type: EmbeddingType, embeds: Dict[int, Tensor]) -> Tensor:
        encoded = {}
        for model_idx, embed in embeds.items():
            if e_type == EmbeddingType.Entity:
                encoded_embed = self.entity_models[model_idx].encode(embed)
            elif e_type == EmbeddingType.Relation:
                encoded_embed = self.relation_models[model_idx].encode(embed)
            else:
                raise ValueError("Unknown embedding type:" + str(e_type))
            encoded[model_idx] = encoded_embed
        res = avg_embeds(encoded)
        return res

    def decode(self, e_type: EmbeddingType, embeds: Tensor) -> Dict[int, Tensor]:
        decoded = {}
        for model_idx in range(self.model_manager.num_models()):
            if e_type == EmbeddingType.Entity:
                decoded_embed = self.entity_models[model_idx].decode(embeds)
            elif e_type == EmbeddingType.Relation:
                decoded_embed = self.relation_models[model_idx].decode(embeds)
            else:
                raise ValueError("Unknown embedding type:" + str(e_type))
            decoded[model_idx] = decoded_embed
        return decoded

    def train_aggregation(self):
        print("Training entity autoencoder")
        self.train_model(EmbeddingType.Entity, self.entity_models)

        print("Training relation autoencoder")
        self.train_model(EmbeddingType.Relation, self.relation_models)

        print("Completed aggregation training.")

    def train_model(self, e_type: EmbeddingType, model):
        model.train()
        # create dataloader
        dataloader = create_aggregation_dataloader(self.model_manager, e_type, 50, True)
        # validation using MSE Loss function
        loss_function = torch.nn.MSELoss()
        # using an Adam Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(self.epochs):
            loss_val = 0
            for _, batch in dataloader:
                encoded_embeds = self.encode(e_type, batch)
                target = self.decode(e_type, encoded_embeds)

                max_emb = 0
                for value in batch.values():
                    max_emb = max(max_emb, value.size()[1])

                emb_in = pad_embeds(batch, max_emb)
                emb_in = concat_embeds(emb_in)
                emb_out = pad_embeds(target, max_emb)
                emb_out = concat_embeds(emb_out)

                loss = loss_function(emb_in, emb_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item()

            self.config.log("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, self.epochs, loss_val))


class Autoencoder(nn.Module, Configurable):
    def __init__(self, config: Config, dim_in, dim_out):
        super(Autoencoder, self).__init__()
        Configurable.__init__(self, config, "autoencoder")

        # set basic autoencoder configuration
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_layers = self.get_option("num_layers")
        self.dropout = self.get_option("dropout")

        # construct model layers
        encode_dict = OrderedDict()
        decode_dict = OrderedDict()

        layer_dims = [
            round(self.dim_in - n * ((self.dim_in - self.dim_out) / self.num_layers)) for n in range(0, self.num_layers)
        ]
        layer_dims.append(self.dim_out)

        i = 0
        for idx in range(0, self.num_layers):
            encode_dict[str(i) + "-dropout"] = nn.Dropout(p=self.dropout)
            i += 1
            encode_dict[str(i) + "-linear"] = nn.Linear(layer_dims[idx], layer_dims[idx + 1], bias=True)
            i += 1
            if idx + 1 < self.num_layers:
                encode_dict[str(i) + "-relu"] = nn.ReLU()
                i += 1
        for idx in reversed(range(0, self.num_layers)):
            decode_dict[str(i) + "-linear"] = nn.Linear(layer_dims[idx + 1], layer_dims[idx], bias=True)
            i += 1
            if idx > 0:
                decode_dict[str(i) + "-relu"] = nn.ReLU()
                i += 1

        self.encoder = torch.nn.Sequential(encode_dict)
        self.decoder = torch.nn.Sequential(decode_dict)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded


class OneToN(AggregationBase):

    def __init__(self, model_manager: ModelManager, dataset: Dataset, config: Config, parent_configuration_key):
        AggregationBase.__init__(self, model_manager, config, "oneton", parent_configuration_key)

        # set model parameters
        self.dataset = dataset

        # create embedders
        self._entity_embedder = torch.nn.Embedding(dataset.num_entities(), self.entity_agg_dim)
        self._relation_embedder = torch.nn.Embedding(dataset.num_relations(), self.relation_agg_dim)

        # create neural nets for metaembedding projection
        num_models = self.model_manager.num_models()
        model_dims = self.model_manager.get_model_dims()
        self.entity_nets = nn.ModuleList(
            [OneToNet(config, self.entity_agg_dim, model_dims[model_idx][EmbeddingType.Entity])
             for model_idx in range(num_models)]
        )
        self.relation_nets = nn.ModuleList(
            [OneToNet(config, self.entity_agg_dim, model_dims[model_idx][EmbeddingType.Relation])
             for model_idx in range(num_models)]
        )

    def aggregate(self, target, indexes: Tensor = None):
        if target == EmbeddingTarget.Subject or target == EmbeddingTarget.Object:
            if indexes is None:
                entity_indexes = torch.arange(
                    self.dataset.num_entities(), dtype=torch.long, device=self._entity_embedder.weight.device
                )
                return self._entity_embedder(entity_indexes)
            else:
                return self._entity_embedder(indexes.long())
        elif target == EmbeddingTarget.Predicate:
            if indexes is None:
                relation_indexes = torch.arange(
                    self.dataset.num_relations(), dtype=torch.long, device=self._entity_embedder.weight.device
                )
                return self._relation_embedder(relation_indexes)
            else:
                return self._relation_embedder(indexes.long())
        else:
            raise ValueError("Unknown target embedding " + str(target))

    def train_aggregation(self):
        pass

    def penalty(self, **kwargs) -> List[Tensor]:
        result = super().penalty(**kwargs)
        penalty_sum = torch.tensor([0], device=self.config.get("job.device"))
        loss_fn = torch.nn.MSELoss()

        # fetch all original model embeddings for entities and relations
        entity_embeds = self.model_manager.fetch_model_embeddings(EmbeddingTarget.Subject)
        relation_embeds = self.model_manager.fetch_model_embeddings(EmbeddingTarget.Predicate)

        # get all entity and relation indexes
        entity_indexes = torch.arange(
            self.dataset.num_entities(), dtype=torch.long, device=self._entity_embedder.weight.device
        )
        relation_indexes = torch.arange(
            self.dataset.num_relations(), dtype=torch.long, device=self._entity_embedder.weight.device
        )

        # compute model specific entity and relation penalties
        for model_idx in range(self.model_manager.num_models()):
            entity_projections = self.entity_nets[model_idx](self._entity_embedder(entity_indexes))
            entity_penalty = loss_fn(entity_embeds[model_idx], entity_projections)
            relation_projections = self.relation_nets[model_idx](self._entity_embedder(relation_indexes))
            relation_penalty = loss_fn(relation_embeds[model_idx], relation_projections)
            penalty_sum += entity_penalty + relation_penalty

        result += [
            (
                f"{self.configuration_key}.projection_penalty",
                penalty_sum
            )
        ]
        return result


class OneToNet(nn.Module, Configurable):

    def __init__(self, config: Config, dim_in, dim_out):
        super(OneToNet, self).__init__()
        Configurable.__init__(self, config, "onetonet")
        self.layer = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x):
        x = self.layer(x)
        return x
