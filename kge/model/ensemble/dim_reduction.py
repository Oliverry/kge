from collections import OrderedDict

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from kge import Configurable, Config


class DimReductionBase(nn.Module, Configurable):

    def __init__(self, config: Config, configuration_key=None):
        Configurable.__init__(self, config, configuration_key)
        nn.Module.__init__(self)
        pass

    def reduce_entities(self, t: Tensor):
        """
        Execute a dimensionality reduction on the tensor of the form n times m times dim_m,
        where n is the number of entities, m is the number of submodels and dim_m is the
        length of the model embedding dimension.
        :param t:
        :return:
        """
        raise NotImplementedError

    def reduce_relations(self, t: Tensor):
        """
        Execute a dimensionality reduction on the tensor of the form n times m times dim_m,
        where n is the number of relations, m is the number of submodels and dim_m is the
        length of the model embedding dimension.
        :param t:
        :return:
        """
        raise NotImplementedError

    def train_dim_reduction(self, models):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self, savepoint):
        raise NotImplementedError


class DimReductionDataset(Dataset):
    def __init__(self, models, mode="entity"):
        """
        Creates a new dataset for unsupervised learning of dimensionality reduction models.
        The data has the format n times m times dim_m.
        :param models:
        :param mode: either "entity" or "relation"
        """
        self.data = None
        for idx, model in enumerate(models):
            m_embeds = None
            if mode == "entity":
                # assuming subject and object embedder are the same
                m_embeds = model.get_s_embedder().embed_all().detach()
            elif mode == "relation":
                m_embeds = model.get_p_embedder().embed_all().detach()
            else:
                raise ValueError
            n = m_embeds.size()[0]
            m_embeds = m_embeds.view(n, 1, -1)
            if self.data is None:
                self.data = m_embeds
            elif self.data is not None:
                self.data = torch.cat((self.data, m_embeds), 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# TODO different configs for entity and relations reduction
class AutoencoderReduction(DimReductionBase):

    def __init__(self, config: Config, configuration_key=None):
        DimReductionBase.__init__(self, config, configuration_key)
        self.entity_model = Autoencoder(config, configuration_key)
        self.relation_model = Autoencoder(config, configuration_key)

    def reduce_entities(self, t: Tensor):
        n = t.size()[0]
        entities = t.view(n, -1)  # transform tensor to format
        entities = self.entity_model.reduce(entities)
        return entities

    def reduce_relations(self, t: Tensor):
        n = t.size()[0]
        relations = t.view(n, -1)  # transform tensor to format
        relations = self.relation_model.reduce(relations)
        return relations

    def train_dim_reduction(self, models):
        # create dataloader
        entity_dataloader = DataLoader(DimReductionDataset(models, "entity"), batch_size=10, shuffle=True)
        relation_dataloader = DataLoader(DimReductionDataset(models, "relation"), batch_size=10, shuffle=True)

        # Validation using MSE Loss function
        loss_function = torch.nn.MSELoss()

        # Using an Adam Optimizer with lr = 0.1
        optimizer = torch.optim.Adam(self.entity_model.parameters(), lr=1e-1, weight_decay=1e-8)
        epochs = 20
        print("Training entity autoencoder")
        for epoch in range(epochs):
            loss_val = 0
            for entity_embeds in entity_dataloader:
                # Reshaping the image to (-1, 784)
                n = entity_embeds.size()[0]
                entities = entity_embeds.view(n, -1)

                # Output of Autoencoder
                reconstructed = self.entity_model(entities)

                # Calculating the loss function
                loss = loss_function(reconstructed, entities)

                # The gradients are set to zero,
                # then the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item()
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss_val))

        # Using an Adam Optimizer with lr = 0.1
        optimizer = torch.optim.Adam(self.relation_model.parameters(), lr=1e-1, weight_decay=1e-8)
        epochs = 20
        print("Training relation autoencoder")
        for epoch in range(epochs):
            loss_val = 0
            for relation_embeds in relation_dataloader:
                # Reshaping the image to (-1, 784)
                n = relation_embeds.size()[0]
                entities = relation_embeds.view(n, -1)

                # Output of Autoencoder
                reconstructed = self.relation_model(entities)

                # Calculating the loss function
                loss = loss_function(reconstructed, entities)

                # The gradients are set to zero,
                # then the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item()
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss_val))

    def save(self):
        state = {"entity": self.entity_model.state_dict(), "relation": self.relation_model.state_dict()}
        return state

    def load(self, savepoint):
        self.entity_model.load_state_dict(savepoint["entity"])
        self.relation_model.load_state_dict(savepoint["relation"])


# TODO add regularization
class Autoencoder(nn.Module, Configurable):
    def __init__(self, config: Config, configuration_key=None):
        super(Autoencoder, self).__init__()
        Configurable.__init__(self, config, configuration_key)
        dim_in = self.get_option("dim_in")
        dim_out = self.get_option("dim_out")
        num_layers = self.get_option("num_layers")

        layer_dims = [round(dim_in - n * ((dim_in - dim_out) / num_layers)) for n in range(0, num_layers)]
        layer_dims.append(dim_out)

        encode_dict = OrderedDict()
        for idx in range(0, num_layers):
            encode_dict["linear" + str(idx)] = nn.Linear(layer_dims[idx], layer_dims[idx + 1])
            if idx + 1 < num_layers:
                encode_dict["relu" + str(idx)] = nn.ReLU()

        decode_dict = OrderedDict()
        for idx in reversed(range(0, num_layers)):
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
        encoded = self.encoder(x)
        return encoded
