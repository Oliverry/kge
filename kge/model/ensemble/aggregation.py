import sys
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
    Computes the concatenation for a given set of embeddings.
    These embeddings are given as a dictionary, where a model index points to a tensor of embeddings.
    Returns a tensor of size n times E, where n is the common number of embeddings and E is the size of the concatenated
    embeddings.
    :param embeds: Dictionary from model index integer to model specific embedding.
    :return: Tensor of concatenated embeddings.
    """
    embeds_list = []
    for model_idx in embeds.keys():
        embeds_list.append(embeds[model_idx])
    out = torch.cat(embeds_list, 1)
    return out


def pad_embeds(embeds, padding_length):
    """
    Applies padding on the given dictionary of embeddings.
    The embeddings are provided as a dictionary pointing from a model index to the model specific embeddings.
    Returns a dictionary, where the model specific embeddings are padded with zeros to the specified length.
    :param embeds: Dictionary of model specific embeddings.
    :param padding_length: Length of the padded model specific embeddings.
    :return: Dictionary of model specific padded embeddings.
    """
    padded_embeds = {}
    for model_idx in embeds:
        model_embeds = embeds[model_idx]
        padded_embed = pad_embed(model_embeds, padding_length)
        padded_embeds[model_idx] = padded_embed
    return padded_embeds


def pad_embed(embed, padding_length):
    """
    Applies padding with zeros on a single tensor of embeddings.
    :param embed: Tensor of embeddings to be padded.
    :param padding_length: Length of the padded embeddings.
    :return: Tensor of padded embeddings.
    """
    last_dim = embed.size()[1]
    padded_embed = F.pad(
        input=embed, pad=(0, padding_length - last_dim), mode="constant", value=0
    )
    return padded_embed


def avg_embeds(embeds):
    """
    Computes the mean for a given set of embeddings.
    These embeddings are given as a dictionary, where a model index points to a tensor of embeddings.
    This tensor has the form n times E, where n is the common number of embeddings and E is the common embedding size.
    Returns a tensor of size n times E.
    :param embeds: Dictionary of model specific embeddings.
    :return: Tensor of average embeddings.
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
    """
    Base class to perform aggregation on a given set of embeddings.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        config: Config,
        configuration_key,
        parent_configuration_key,
    ):
        Configurable.__init__(self, config, configuration_key)
        nn.Module.__init__(self)
        self.model_manager = model_manager

        # compute aggregated entity and relation dimensions
        self.entity_agg_dim = 0
        self.relation_agg_dim = 0
        self.compute_dims()

        # write agg dims for evaluator in embedding ensemble config
        self.config.set(
            parent_configuration_key + ".entity_agg_dim", self.entity_agg_dim
        )
        self.config.set(
            parent_configuration_key + ".relation_agg_dim", self.relation_agg_dim
        )

    def compute_dims(self):
        """
        Computes the entity and relation embedding dimensions after aggregation has been applied.
        For this the reduction option in the config for the respective aggregation method are considered.
        :return:
        """
        # compute concatenated agg dim
        embed_dims = self.model_manager.get_model_dims()
        embed_dim = 0
        for model_idx in range(self.model_manager.num_models()):
            embed_dim += embed_dims[model_idx]

        # get entity and relation reduction by percentage
        entity_reduction = 1.0
        relation_reduction = 1.0
        if self.has_option("entity_reduction"):
            entity_reduction = self.get_option("entity_reduction")
        if self.has_option("relation_reduction"):
            relation_reduction = self.get_option("relation_reduction")

        # compute entity dim
        self.entity_agg_dim = round(embed_dim * entity_reduction)
        # check if entity dim is even and correct it (required by some models)
        if self.entity_agg_dim % 2 != 0:
            self.entity_agg_dim += 1
        # compute relation dimension on base of entity embedding size
        self.relation_agg_dim = round(
            self.entity_agg_dim * (relation_reduction / entity_reduction)
        )

    def aggregate(self, target: EmbeddingTarget, indexes: Tensor = None):
        """
        Applies an aggregation function on a set of embeddings to compute a metaembedding.
        The embeddings are fetched first using the model manager for a given embedding target.
        If the indexes are given, they are used as IDs for entities or relations, otherwise all known entities or
        relations are fetched.
        :param target: Specifies whether entities or relations shall be combined.
        :param indexes: Tensor of entity or relation IDs.
        :return: Aggregated embeddings of all base models
        """
        raise NotImplementedError

    def train_aggregation(self):
        """
        Executes an optional optimization routine for the aggregation methods.
        :return:
        """
        raise NotImplementedError

    def penalty(self, **kwargs) -> List[Tensor]:
        return []


class Concat(AggregationBase):
    """
    Aggregation method that combines the embeddings by simple concatenation.
    """

    def __init__(self, model_manager: ModelManager, config, parent_configuration_key):
        AggregationBase.__init__(
            self, model_manager, config, "concat", parent_configuration_key
        )

    def aggregate(self, target: EmbeddingTarget, indexes: Tensor = None):
        t = self.model_manager.fetch_model_embeddings(target, indexes)
        res = concat_embeds(t)
        res = pad_embed(res, self.entity_agg_dim)
        return res

    def train_aggregation(self):
        pass


class MeanReduction(AggregationBase):
    """
    This class computes the metaembeddings by taking the mean over all base model embeddings.
    """

    def __init__(self, model_manager: ModelManager, config, parent_configuration_key):
        AggregationBase.__init__(
            self, model_manager, config, "mean", parent_configuration_key
        )

    def compute_dims(self):
        # find the longest entity and relation embeddings
        embed_dims = self.model_manager.get_model_dims()
        for model_idx in range(self.model_manager.num_models()):
            self.entity_agg_dim = max(self.entity_agg_dim, embed_dims[model_idx])

        # check if embedding dim is even (required by some models)
        if self.entity_agg_dim % 2 != 0:
            self.entity_agg_dim += 1

        # in case of mean relations and embedding dim are equal
        self.relation_agg_dim = self.entity_agg_dim

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
    """
    Applies PCA on the set of entities and relation to be used for metaembeddings.
    """

    def __init__(
        self, model_manager: ModelManager, dataset, config, parent_configuration_key
    ):
        AggregationBase.__init__(
            self, model_manager, config, "pca", parent_configuration_key
        )

        # contain the metaembeddings for entities and relations
        self._entity_embedder = torch.nn.Embedding(
            dataset.num_entities(), self.entity_agg_dim
        )
        self._relation_embedder = torch.nn.Embedding(
            dataset.num_relations(), self.relation_agg_dim
        )

    def aggregate(self, target: EmbeddingTarget, indexes: Tensor = None):
        if target == EmbeddingTarget.Subject or target == EmbeddingTarget.Object:
            if indexes is None:
                entity_indexes = torch.arange(
                    self.dataset.num_entities(),
                    dtype=torch.long,
                    device=self._entity_embedder.weight.device,
                )
                return self._entity_embedder(entity_indexes)
            else:
                return self._entity_embedder(indexes.long())
        elif target == EmbeddingTarget.Predicate:
            if indexes is None:
                relation_indexes = torch.arange(
                    self.dataset.num_relations(),
                    dtype=torch.long,
                    device=self._entity_embedder.weight.device,
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
        entity_embeds = self.model_manager.fetch_model_embeddings(
            EmbeddingTarget.Subject
        )
        relation_embeds = self.model_manager.fetch_model_embeddings(
            EmbeddingTarget.Predicate
        )
        entity_embeds = concat_embeds(entity_embeds)
        relation_embeds = concat_embeds(relation_embeds)

        # move to cpu for scikit-learn
        entity_embeds = entity_embeds.to(torch.device("cpu"))
        relation_embeds = relation_embeds.to(torch.device("cpu"))

        # prior standardization
        entity_embeds = StandardScaler().fit_transform(entity_embeds)
        relation_embeds = StandardScaler().fit_transform(relation_embeds)

        # apply pca
        entity_embeds = entity_pca.fit_transform(entity_embeds)
        entity_embeds = torch.tensor(
            entity_embeds, dtype=torch.float, device=self.config.get("job.device")
        )
        relation_embeds = relation_pca.fit_transform(relation_embeds)
        relation_embeds = torch.tensor(
            relation_embeds, dtype=torch.float, device=self.config.get("job.device")
        )

        # store embeddings
        self._entity_embedder = nn.Embedding.from_pretrained(entity_embeds)
        self._relation_embedder = nn.Embedding.from_pretrained(relation_embeds)


class AutoencoderReduction(AggregationBase):
    """
    Applies the AAEME method to compute metaembeddings.
    """

    def __init__(
        self, model_manager: ModelManager, config: Config, parent_configuration_key
    ):
        AggregationBase.__init__(
            self,
            model_manager,
            config,
            "autoencoder_reduction",
            parent_configuration_key,
        )

        # get training parameters
        self.epochs = self.get_option("epochs")
        self.lr = self.get_option("lr")
        self.batch_size = self.get_option("batch_size")
        self.patience = self.get_option("patience")

        # create autoencoders
        entity_autoencoders = []
        relation_autoencoders = []
        embed_dims = self.model_manager.get_model_dims()
        for model_idx in range(self.model_manager.num_models()):
            dim_in = embed_dims[model_idx]
            entity_ae = Autoencoder(config, dim_in, self.entity_agg_dim)
            entity_ae.to(config.get("job.device"))
            entity_autoencoders.append(entity_ae)
            relation_ae = Autoencoder(config, dim_in, self.relation_agg_dim)
            relation_ae.to(config.get("job.device"))
            relation_autoencoders.append(relation_ae)
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
        """
        Given a dictionary of embeddings, encode them with their model specific Autoencoder.
        :param e_type: Specifies whether entities or relation shall be processed.
        :param embeds: Dictionary of model-specific embeddings.
        :return: Dictionary of model-specific encoded embeddings.
        """
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
        """
        Given a dictionary of embeddings, decode them with their model specific Autoencoder.
        :param e_type: Specifies whether entities or relation shall be processed.
        :param embeds: Dictionary of model-specific encoded embeddings.
        :return: Dictionary of model-specific decoded embeddings.
        """
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
        self.config.log("Training entity autoencoder")
        self.train_model(EmbeddingType.Entity, self.entity_models)

        self.config.log("Training relation autoencoder")
        self.train_model(EmbeddingType.Relation, self.relation_models)

        self.config.log("Completed aggregation training.")

    def train_model(self, e_type: EmbeddingType, model):
        """
        Subprocess to train entitiy and relation Autoencoder separately.
        :param e_type: Specifies whether entity or relations Autoencoder shall be optimized.
        :param model: Set of Autoencoders to be optimized.
        :return:
        """
        # create dataloader
        dataloader_train, dataloader_valid = create_aggregation_dataloader(
            self.model_manager, e_type, 0.8, self.batch_size, True
        )
        # using MSE Loss function
        loss_function = torch.nn.MSELoss()
        # using an Adam Optimizer
        optimizer = torch.optim.Adagrad(model.parameters(), lr=self.lr)

        # Early stopping
        last_loss = sys.maxsize
        trigger_times = 0

        for epoch in range(self.epochs):
            model.train()
            loss_val = 0
            for _, batch in dataloader_train:
                # encode and decode input
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

            # Early stopping
            current_loss = self.validate(model, e_type, dataloader_valid, loss_function)

            if current_loss > last_loss:
                trigger_times += 1
                if trigger_times >= self.patience:
                    return
            else:
                trigger_times = 0
            last_loss = current_loss

            self.config.log(
                "epoch : {}/{}, loss = {:.6f}, valid = {:.6f}".format(
                    epoch + 1, self.epochs, loss_val, current_loss
                )
            )

    def validate(self, model, e_type, dataloader_valid, loss_function):
        """
        Separate validation function for aggregation optimization to capture the performance on the validation dataset.
        :param model: Autoencoder models to be validated.
        :param e_type: Specifies whether entities or relation will be validated.
        :param dataloader_valid: The dataloader with the validation dataset.
        :param loss_function: The loss function, which shall be used for validation.
        :return:
        """
        model.eval()
        loss_total = 0

        # Test validation data
        with torch.no_grad():
            for _, batch in dataloader_valid:
                # encode and decode input
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
                loss_total += loss.item()

        return loss_total / len(dataloader_valid)


class Autoencoder(nn.Module, Configurable):
    """
    A single Autoenocder to encode and decode embeddings for specified embedding sizes.
    """

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

        # compute layer dimensions
        layer_dims = [
            round(self.dim_in - n * ((self.dim_in - self.dim_out) / self.num_layers))
            for n in range(0, self.num_layers)
        ]
        layer_dims.append(self.dim_out)

        # create autoencoder layers
        i = 0
        for idx in range(0, self.num_layers):
            encode_dict[str(i) + "-dropout"] = nn.Dropout(p=self.dropout)
            i += 1
            encode_dict[str(i) + "-linear"] = nn.Linear(
                layer_dims[idx], layer_dims[idx + 1], bias=True
            )
            i += 1
            if idx + 1 < self.num_layers:
                encode_dict[str(i) + "-relu"] = nn.ReLU()
                i += 1
        for idx in reversed(range(0, self.num_layers)):
            decode_dict[str(i) + "-linear"] = nn.Linear(
                layer_dims[idx + 1], layer_dims[idx], bias=True
            )
            i += 1
            if idx > 0:
                decode_dict[str(i) + "-relu"] = nn.ReLU()
                i += 1

        self.encoder = torch.nn.Sequential(encode_dict)
        self.decoder = torch.nn.Sequential(decode_dict)

    def forward(self, x):
        """
        Applies an encoding and decoding of embeddings.
        :param x: The embeddings to be encoded and decoded.
        :return: The decoded embeddings.
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def encode(self, x):
        """
        Encodes the given embeddings.
        :param x: The embeddings to be encoded.
        :return: The encoded embeddings.
        """
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        """
        Decodes the given embeddings.
        :param x: The encoded embeddings to be decoded.
        :return: The decoded embeddings.
        """
        decoded = self.decoder(x)
        return decoded


class OneToN(AggregationBase):
    """
    This class applies the one to n aggregation technique.
    It was implemented for end to end learning in a given embedding ensemble model.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        dataset: Dataset,
        config: Config,
        parent_configuration_key,
    ):
        AggregationBase.__init__(
            self, model_manager, config, "oneton", parent_configuration_key
        )

        # set model parameters
        self.dataset = dataset
        mean = self.get_option("mean")
        std = self.get_option("std")

        # create embedders
        self._entity_embedder = torch.nn.Embedding(
            dataset.num_entities(), self.entity_agg_dim
        )
        self._relation_embedder = torch.nn.Embedding(
            dataset.num_relations(), self.relation_agg_dim
        )
        torch.nn.init.normal_(self._entity_embedder.weight, mean=mean, std=std)
        torch.nn.init.normal_(self._relation_embedder.weight, mean=mean, std=std)

        # create neural nets for metaembedding projection
        num_models = self.model_manager.num_models()
        model_dims = self.model_manager.get_model_dims()
        self.entity_nets = nn.ModuleList(
            [
                OneToNet(config, self.entity_agg_dim, model_dims[model_idx])
                for model_idx in range(num_models)
            ]
        )
        self.relation_nets = nn.ModuleList(
            [
                OneToNet(config, self.entity_agg_dim, model_dims[model_idx])
                for model_idx in range(num_models)
            ]
        )

    def aggregate(self, target, indexes: Tensor = None):
        if target == EmbeddingTarget.Subject or target == EmbeddingTarget.Object:
            if indexes is None:
                entity_indexes = torch.arange(
                    self.dataset.num_entities(),
                    dtype=torch.long,
                    device=self._entity_embedder.weight.device,
                )
                return self._entity_embedder(entity_indexes)
            else:
                return self._entity_embedder(indexes.long())
        elif target == EmbeddingTarget.Predicate:
            if indexes is None:
                relation_indexes = torch.arange(
                    self.dataset.num_relations(),
                    dtype=torch.long,
                    device=self._entity_embedder.weight.device,
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
        penalty_sum = torch.tensor(
            [0], dtype=torch.float32, device=self.config.get("job.device")
        )
        loss_fn = torch.nn.MSELoss()

        # fetch all original model embeddings for entities and relations
        entity_embeds = self.model_manager.fetch_model_embeddings(
            EmbeddingTarget.Subject
        )
        relation_embeds = self.model_manager.fetch_model_embeddings(
            EmbeddingTarget.Predicate
        )

        # get all entity and relation indexes
        entity_indexes = torch.arange(
            self.dataset.num_entities(),
            dtype=torch.long,
            device=self._entity_embedder.weight.device,
        )
        relation_indexes = torch.arange(
            self.dataset.num_relations(),
            dtype=torch.long,
            device=self._entity_embedder.weight.device,
        )

        # compute model specific entity and relation penalties
        for model_idx in range(self.model_manager.num_models()):
            entity_projections = self.entity_nets[model_idx](
                self._entity_embedder(entity_indexes)
            )
            entity_penalty = loss_fn(entity_embeds[model_idx], entity_projections)
            relation_projections = self.relation_nets[model_idx](
                self._entity_embedder(relation_indexes)
            )
            relation_penalty = loss_fn(relation_embeds[model_idx], relation_projections)
            penalty_sum += entity_penalty + relation_penalty

        result += [(f"{self.configuration_key}.projection_penalty", penalty_sum)]
        return result


class OneToNet(nn.Module, Configurable):
    """
    A single neural network to project model-specific metaembeddings to a single source embedding set.
    """

    def __init__(self, config: Config, dim_in, dim_out):
        super(OneToNet, self).__init__()
        Configurable.__init__(self, config, "onetonet")
        dropout = self.get_option("dropout")
        self.layer = nn.Linear(dim_in, dim_out, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Project the given embeddings to their source embeddings.
        :param x: The metaembeddings to be projected.
        :return: The transformed source embeddings.
        """
        x = self.dropout(x)
        x = self.layer(x)
        return x
