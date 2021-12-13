from collections import OrderedDict

import torch
from torch import nn, Tensor

from kge import Configurable, Config


class DimReductionBase(Configurable):

    def __init__(self, config: Config, configuration_key=None):
        super(DimReductionBase, self).__init__(config, configuration_key)
        pass

    def reduce_entities(self, t: Tensor):
        """
        Execute a dimensionality reduction on the tensor of the form n times m times dim_m,
        where n is the number of entities, m is the number of submodels and dim_m is the specific
        length of the model embedding dimension.
        :param t:
        :return:
        """
        raise NotImplementedError

    def reduce_relations(self, t: Tensor):
        """
        Execute a dimensionality reduction on the tensor of the form n times m times dim_m,
        where n is the number of relations, m is the number of submodels and dim_m is the specific
        length of the model embedding dimension.
        :param t:
        :return:
        """
        raise NotImplementedError


class AutoencoderReduction(DimReductionBase):

    def __init__(self, config: Config, configuration_key=None):
        super(DimReductionBase, self).__init__(config, configuration_key)
        self.entity_model = Autoencoder(config, configuration_key)
        self.relation_model = Autoencoder(config, configuration_key)

    def reduce_entities(self, t: Tensor):
        pass

    def reduce_relations(self, t: Tensor):
        pass


class Autoencoder(nn.Module, Configurable):
    def __init__(self, config: Config, configuration_key=None):
        super(Autoencoder, self).__init__()
        Configurable.__init__(self, config, configuration_key)
        dim_in = 100  # self.get_option("dim_in")
        dim_out = 50  # self.get_option("dim_out")
        num_layers = 3  # self.get_option("num_layers")

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
                encode_dict["relu" + str(idx)] = nn.ReLU()

        self.encoder = torch.nn.Sequential(encode_dict)
        self.decoder = torch.nn.Sequential(decode_dict)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reduce(self, x):
        reduced = self.encoder(x)
        return reduced
