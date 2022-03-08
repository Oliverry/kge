from collections import OrderedDict
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn, Tensor, optim
from torch.utils.data import Dataset

from kge import Configurable, Config
from kge.model import KgeEmbedder
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
            out = self._entity_embedder(indexes.long())
        elif target == EmbeddingTarget.Predicate:
            out = self._relation_embedder(indexes.long())
        else:
            raise ValueError("Unknown target embedding:" + str(target))
        return out

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
        relation_embeds = relation_pca.fit_transform(relation_embeds)

        # store embeddings
        self._entity_embedder = nn.Embedding.from_pretrained(torch.tensor(entity_embeds, dtype=torch.float))
        self._relation_embedder = nn.Embedding.from_pretrained(torch.tensor(relation_embeds, dtype=torch.float))


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
        return res

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
        for model_idx in self.model_manager.num_models():
            if e_type == EmbeddingType.Entity:
                decoded_embed = self.entity_models[model_idx].decode(embeds)
            elif e_type == EmbeddingType.Relation:
                decoded_embed = self.relation_models[model_idx].decode(embeds)
            else:
                raise ValueError("Unknown embedding type:" + str(e_type))
            decoded[model_idx] = decoded_embed
        return decoded

    def train_aggregation(self):
        # create dataloader
        entity_dataloader = create_aggregation_dataloader(self.model_manager, EmbeddingType.Entity, 50, True)
        relation_dataloader = create_aggregation_dataloader(self.model_manager, EmbeddingType.Relation, 50, True)

        print("Training entity autoencoder")
        self.train_model(entity_dataloader, self.entity_models)

        print("Training relation autoencoder")
        self.train_model(relation_dataloader, self.relation_models)

        print("Completed aggregation training.")

    def train_model(self, dataloader, model):
        model.train()
        # validation using MSE Loss function
        loss_function = torch.nn.MSELoss()
        # using an Adam Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(self.epochs):
            loss_val = 0
            for _, batch in dataloader:
                n = batch.size()[0]
                embeds = batch.view(n, -1)

                # Output of Autoencoder
                reconstructed = model(embeds)

                # Calculating the loss function
                loss = loss_function(reconstructed, embeds)

                # The gradients are set to zero,
                # then the gradient is computed and stored.
                # .step() performs parameter update
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

    def __init__(self, model_manager: ModelManager, dataset: Dataset, config: Config,
                 parent_configuration_key, init_for_load_only=False):
        AggregationBase.__init__(self, model_manager, config, "oneton", parent_configuration_key)

        # modify embedder config
        entity_dim = config.get(parent_configuration_key + ".entities.agg_dim")
        relation_dim = config.get(parent_configuration_key + ".relations.agg_dim")
        num_models = len(config.get(parent_configuration_key + ".base_models"))
        self.set_option("entity_embedder.dim", entity_dim * num_models)
        self.set_option("relation_embedder.dim", relation_dim * num_models)

        # create embedders for metaembeddings
        # TODO if init_for_load_only, load embeddings?
        num_models = len(config.get(parent_configuration_key + ".base_models"))
        self._entity_embedder = KgeEmbedder.create(
            config,
            dataset,
            self.configuration_key + ".entity_embedder",
            dataset.num_entities(),
            init_for_load_only=init_for_load_only,
        )
        self._relation_embedder = KgeEmbedder.create(
            config,
            dataset,
            self.configuration_key + ".relation_embedder",
            dataset.num_relations(),
            init_for_load_only=init_for_load_only,
        )

        # create neural nets for metaembedding projection
        # TODO do not load nets when evaluation?
        self.entity_nets = nn.ModuleList(
            [OneToNet(config, parent_configuration_key, "entities") for _ in range(0, num_models)]
        )
        self.relations_nets = nn.ModuleList(
            [OneToNet(config, parent_configuration_key, "relations") for _ in range(0, num_models)]
        )

        # get training parameters if training
        if not init_for_load_only:
            self.epochs = self.get_option("epochs")

    def aggregate(self, target, indexes: Tensor = None):
        if target == "s" or target == "o":
            if indexes is None:
                return self._entity_embedder.embed_all()
            else:
                return self._entity_embedder.embed(indexes)
        elif target == "p":
            if indexes is None:
                return self._relation_embedder.embed_all()
            else:
                return self._relation_embedder.embed(indexes)
        else:
            raise ValueError("Unknown target embedding.")

    def train_aggregation(self):
        # create dataloader
        entity_dataloader = create_aggregation_dataloader(self.model_manager, "entities", 50, True)
        relation_dataloader = create_aggregation_dataloader(self.model_manager, "relations", 50, True)

        print("Training entity autoencoder")
        self.train_model(entity_dataloader, self.entity_nets, self._entity_embedder)

        print("Training relation autoencoder")
        self.train_model(relation_dataloader, self.relations_nets, self._relation_embedder)

    def train_model(self, dataloader, models, embedder):
        # Define the loss
        loss_func = nn.MSELoss()
        optimizer = optim.SGD(models.parameters(), lr=0.1)
        for e in range(self.epochs):
            running_loss = 0
            for indexes, data in dataloader:
                # zero the parameter gradients
                optimizer.zero_grad()

                for m, net in enumerate(models):
                    model_embeds = data[:, m]
                    out = net(embedder.embed(indexes))
                    loss = loss_func(out, model_embeds)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                # normalization of embedders
            else:
                print(f"Training loss: {running_loss / len(dataloader)}")


class OneToNet(nn.Module, Configurable):

    def __init__(self, config: Config, parent_configuration_key, embedding_key):
        super(OneToNet, self).__init__()
        Configurable.__init__(self, config, "onetonet")
        num_models = len(config.get(parent_configuration_key + ".base_models"))
        source_dim = config.get(parent_configuration_key + "." + embedding_key + ".source_dim")
        reduced_dim = config.get(parent_configuration_key + "." + embedding_key + ".agg_dim")
        self.layer = nn.Linear(reduced_dim * num_models, source_dim, bias=False)

    def forward(self, x):
        x = self.layer(x)
        return x
