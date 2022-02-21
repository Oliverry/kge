import math
from collections import OrderedDict

import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn, Tensor, optim
from torch.utils.data import Dataset, DataLoader

from kge import Configurable, Config
from kge.model import KgeEmbedder
from kge.model.ensemble.aggregation_data import AggregationDataset, create_aggregation_dataloader

# TODO change embedding fetching
from kge.model.ensemble.model_manager import ModelManager


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

        # compute and set aggregated dimension
        if config.get(parent_configuration_key + ".entities.agg_dim") < 0:
            entity_dim = config.get(parent_configuration_key + ".entities.source_dim")
            entity_reduction = 1.0
            if self.has_option("entity_reduction"):
                entity_reduction = self.get_option("entity_reduction")
            entity_agg = round(entity_dim * entity_reduction)
            config.set(parent_configuration_key + ".entities.agg_dim", entity_agg)
        if config.get(parent_configuration_key + ".relations.agg_dim") < 0:
            relation_dim = config.get(parent_configuration_key + ".relations.source_dim")
            relation_reduction = 1.0
            if self.has_option("relation_reduction"):
                relation_reduction = self.get_option("relation_reduction")
            relation_agg = round(relation_dim * relation_reduction)
            config.set(parent_configuration_key + ".relations.agg_dim", relation_agg)

    def aggregate(self, target, indexes: Tensor = None):
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
        AggregationBase.__init__(self, model_manager, config, None, parent_configuration_key)

    def aggregate(self, target, indexes: Tensor = None):
        t = self.model_manager.fetch_model_embeddings(target, indexes)
        n = t.size()[0]
        res = t.view(n, -1)
        return res

    def train_aggregation(self):
        pass


class PcaReduction(AggregationBase):

    def __init__(self, model_manager: ModelManager, dataset, config, parent_configuration_key):
        AggregationBase.__init__(self, model_manager, config, "pca", parent_configuration_key)

        num_models = len(config.get(parent_configuration_key + ".base_models"))
        self.entity_dim = config.get(parent_configuration_key + ".entities.agg_dim") * num_models
        self.relation_dim = config.get(parent_configuration_key + ".relations.agg_dim") * num_models

        self._entity_embedder = torch.nn.Embedding(dataset.num_entities(), self.entity_dim)
        self._relation_embedder = torch.nn.Embedding(dataset.num_relations(), self.relation_dim)

    def aggregate(self, target, indexes: Tensor = None):
        if target == "s" or target == "o":
            out = self._entity_embedder(indexes.long())
            return out
        elif target == "p":
            out = self._relation_embedder(indexes.long())
            return out
        else:
            raise ValueError("Unknown target embedding.")

    def train_aggregation(self):
        # create pca models
        entity_pca = PCA(n_components=self.entity_dim)
        relation_pca = PCA(n_components=self.relation_dim)

        # fetch and preprocess embeddings
        entity_embeds = self.model_manager.fetch_model_embeddings("s")
        relation_embeds = self.model_manager.fetch_model_embeddings("p")
        n = entity_embeds.size()[0]
        m = relation_embeds.size()[0]
        entity_embeds = entity_embeds.view(n, -1)
        relation_embeds = relation_embeds.view(m, -1)

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

        if config.get("job.type") == "train":
            self.epochs = self.get_option("epochs")
            self.lr = self.get_option("lr")
            self.weight_decay = self.get_option("weight_decay")

        self.entity_model = Autoencoder(config, parent_configuration_key, "entities")
        self.relation_model = Autoencoder(config, parent_configuration_key, "relations")

    def aggregate(self, target, indexes: Tensor = None):
        embeds = self.model_manager.fetch_model_embeddings(target, indexes)
        n = embeds.size()[0]
        embeds = embeds.view(n, -1)  # transform tensor to autoencoder format
        if target == "s" or target == "o":
            embeds = self.entity_model.reduce(embeds)
        elif target == "p":
            embeds = self.relation_model.reduce(embeds)
        return embeds

    def train_aggregation(self):
        # create dataloader
        entity_dataloader = create_aggregation_dataloader(self.model_manager, "entities", 50, True)
        relation_dataloader = create_aggregation_dataloader(self.model_manager, "relations", 50, True)

        print("Training entity autoencoder")
        self.train_model(entity_dataloader, self.entity_model)

        print("Training relation autoencoder")
        self.train_model(relation_dataloader, self.relation_model)

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
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, self.epochs, loss_val))


class Autoencoder(nn.Module, Configurable):
    def __init__(self, config: Config, parent_configuration_key, embedding_key):
        super(Autoencoder, self).__init__()
        Configurable.__init__(self, config, "autoencoder")

        # get basic model configuration
        num_models = len(config.get(parent_configuration_key + ".base_models"))
        source_dim = config.get(parent_configuration_key + "." + embedding_key + ".source_dim")
        reduced_dim = config.get(parent_configuration_key + "." + embedding_key + ".agg_dim")
        self.dim_in = num_models * source_dim
        self.dim_out = num_models * reduced_dim
        self.num_layers = self.get_option("num_layers")
        self.dropout = self.get_option("dropout")

        # construct model layers and combine them
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reduce(self, x):
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(x)
        return encoded


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
