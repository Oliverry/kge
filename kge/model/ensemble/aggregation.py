import math
from collections import OrderedDict

import torch
from sklearn.decomposition import PCA
from torch import nn, Tensor, optim
from torch.utils.data import Dataset, DataLoader

from kge import Configurable, Config
from kge.model import KgeEmbedder
from kge.model.ensemble.aggregation_data import AggregationDataset, fetch_model_embeddings


# TODO change embedding fetching
class AggregationBase(nn.Module, Configurable):

    def __init__(self, models, config: Config, configuration_key, parent_configuration_key):
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
        self.models = models

        # update reduced embedding size in config in case of training
        if config.get("job.type") == "train":
            parent_configuration_key = parent_configuration_key
            num_models = len(config.get(parent_configuration_key + ".submodels"))
            entity_dim = config.get(parent_configuration_key + ".entities.source_dim")
            relation_dim = config.get(parent_configuration_key + ".relations.source_dim")
            entity_reduction = 1.0
            if self.has_option("entity_reduction"):
                entity_reduction = self.get_option("entity_reduction")
            relation_reduction = 1.0
            if self.has_option("relation_reduction"):
                relation_reduction = self.get_option("relation_reduction")
            num_rmm = config.get(parent_configuration_key + ".num_rrm")
            entity_agg = math.floor(num_models * entity_dim * entity_reduction)
            relation_agg = math.floor((num_models + num_rmm) * relation_dim * relation_reduction)
            if self.config.get(parent_configuration_key + ".entities.agg_dim") < 0:
                self.config.set(parent_configuration_key + ".entities.agg_dim", entity_agg)
            if self.config.get(parent_configuration_key + ".relations.agg_dim") < 0:
                self.config.set(parent_configuration_key + ".relations.agg_dim", relation_agg)

    def aggregate(self, target, indexes: Tensor = None):
        """
        Applies an aggregation function for the given indexes and the specified target of
        embedder
        :param target: Can have the values "s", "p" and "o"
        :param indexes: Tensor of entity or relation indexes
        :return: Aggregated embeddings of multiple models
        """
        raise NotImplementedError

    def train_aggregation(self, models):
        raise NotImplementedError


class Concatenation(AggregationBase):

    def __init__(self, models, config, parent_configuration_key):
        AggregationBase.__init__(self, models, config, None, parent_configuration_key)

    def aggregate(self, target, indexes: Tensor = None):
        t = fetch_model_embeddings(self.models, target, indexes)
        n = t.size()[0]
        res = t.view(n, -1)
        return res

    def train_aggregation(self, models):
        pass


class PcaReduction(AggregationBase):

    def __init__(self, models, config, parent_configuration_key):
        AggregationBase.__init__(self, models, config, "pca", parent_configuration_key)
        entity_source = config.get(parent_configuration_key + ".entities.source_dim")
        relation_source = config.get(parent_configuration_key + ".relations.source_dim")
        entity_dim = entity_source * self.get_option("entity_reduction")
        relation_dim = relation_source * self.get_option("relation_reduction")
        if entity_dim - int(entity_dim) > 0:
            raise Exception("PCA reduction of entities not doable for given percentage.")
        if relation_dim - int(relation_dim) > 0:
            raise Exception("PCA reduction of relations not doable for given percentage.")
        self.entity_pca = PCA(n_components=int(entity_dim))
        self.relation_pca = PCA(n_components=int(relation_dim))

    def aggregate(self, target, indexes: Tensor = None):
        t = torch.randn(2, 128)
        self.entity_pca.fit(t)
        res = self.entity_pca.transform(t)
        return res

    def train_aggregation(self, models):
        pass


class AutoencoderReduction(AggregationBase):

    def __init__(self, models, config: Config, parent_configuration_key):
        AggregationBase.__init__(self, models, config, "autoencoder_reduction", parent_configuration_key)

        if config.get("job.type") == "train":
            self.epochs = self.get_option("epochs")
            self.lr = self.get_option("lr")
            self.weight_decay = self.get_option("weight_decay")

        self.entity_model = Autoencoder(config, parent_configuration_key, "entities")
        self.relation_model = Autoencoder(config, parent_configuration_key, "relations")

    def aggregate(self, target, indexes: Tensor = None):
        t = fetch_model_embeddings(self.models, target, indexes)
        n = t.size()[0]
        embeds = t.view(n, -1)  # transform tensor to autoencoder format
        if target == "s" or target == "o":
            embeds = self.entity_model.reduce(t)
        elif target == "p":
            embeds = self.relation_model.reduce(t)
        return embeds

    def train_aggregation(self, models):
        # create dataloader
        s_embs = self.ensemble.fetch_model_embeddings("s")
        p_embs = self.ensemble.fetch_model_embeddings("p")
        entity_dataloader = DataLoader(AggregationDataset(s_embs), batch_size=10, shuffle=True)
        relation_dataloader = DataLoader(AggregationDataset(p_embs), batch_size=10, shuffle=True)

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
            for batch in dataloader:
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
    def __init__(self, config: Config, parent_configuration_key, embedding_configuration_key):
        super(Autoencoder, self).__init__()
        Configurable.__init__(self, config, "autoencoder")
        num_models = len(config.get(parent_configuration_key + ".submodels"))
        source_dim = config.get(parent_configuration_key + "." + embedding_configuration_key + ".source_dim")
        reduced_dim = config.get(parent_configuration_key + "." + embedding_configuration_key + ".agg_dim")
        self.dim_in = num_models * source_dim
        self.dim_out = reduced_dim
        self.num_layers = self.get_option("num_layers")
        self.dropout = self.get_option("dropout")

        layer_dims = [
            round(self.dim_in - n * ((self.dim_in - self.dim_out) / self.num_layers)) for n in range(0, self.num_layers)
        ]
        layer_dims.append(self.dim_out)

        encode_dict = OrderedDict()
        for idx in range(0, self.num_layers):
            encode_dict["dropout" + str(idx)] = nn.Dropout(p=self.dropout)
            encode_dict["linear" + str(idx)] = nn.Linear(layer_dims[idx], layer_dims[idx + 1])
            if idx + 1 < self.num_layers:
                encode_dict["relu" + str(idx)] = nn.ReLU()

        decode_dict = OrderedDict()
        for idx in reversed(range(0, self.num_layers)):
            decode_dict["linear" + str(idx)] = nn.Linear(layer_dims[idx + 1], layer_dims[idx])
            if idx > 0:
                decode_dict["relu" + str(idx)] = nn.ReLU()

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

    def __init__(self, models, dataset: Dataset, config: Config, parent_configuration_key, init_for_load_only=False):
        AggregationBase.__init__(self, models, config, "oneton", parent_configuration_key)

        # modify embedder config
        entity_dim = config.get(parent_configuration_key + ".entities.agg_dim")
        relation_dim = config.get(parent_configuration_key + ".relations.agg_dim")
        self.set_option("entity_embedder.dim", entity_dim)
        self.set_option("relation_embedder.dim", relation_dim)

        # create embedders for metaembeddings
        # TODO if init_for_load_only, load embeddings?
        num_models = len(config.get(parent_configuration_key + ".submodels"))
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
        if target == "p":
            if indexes is None:
                return self._relation_embedder.embed_all()
            else:
                return self._relation_embedder.embed(indexes)
        else:
            if indexes is None:
                return self._entity_embedder.embed_all()
            else:
                return self._entity_embedder.embed(indexes)

    def train_aggregation(self, models):
        # create dataloader
        s_embs = fetch_model_embeddings(self.models, "s")
        p_embs = fetch_model_embeddings(self.models, "p")
        entity_dataloader = DataLoader(AggregationDataset(s_embs), batch_size=10, shuffle=True)
        relation_dataloader = DataLoader(AggregationDataset(p_embs), batch_size=10, shuffle=True)

        print("Training entity autoencoder")
        self.train_model(entity_dataloader, self.entity_nets, self._entity_embedder)

        print("Training relation autoencoder")
        self.train_model(relation_dataloader, self.relations_nets, self._relation_embedder)

    def train_model(self, dataloader, models, embedder):
        # Define the loss
        loss_func = nn.MSELoss()
        optimizer = optim.SGD(models.parameters(), lr=0.003)
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

    def __init__(self, config: Config, parent_configuration_key, embedding_configuration_key):
        super(OneToNet, self).__init__()
        Configurable.__init__(self, config, "onetonet")
        source_dim = config.get(parent_configuration_key + "." + embedding_configuration_key + ".source_dim")
        reduced_dim = config.get(parent_configuration_key + "." + embedding_configuration_key + ".agg_dim")
        self.layer = nn.Linear(reduced_dim, source_dim, bias=False)

    def forward(self, x):
        x = self.layer(x)
        return x
