import torch
from torch.autograd import Variable
from torch.autograd.grad_mode import F

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class ComplexLiteraleScorer(RelationalScorer):

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

        # from config
        num_entities = 0
        embedding_dim = 0
        numerical_literals = 0
        num_relations = 0
        input_dropout = 0

        self.emb_dim = embedding_dim

        self.emb_e_real = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit_real = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim + self.n_num_lit, self.emb_dim),
            torch.nn.Tanh()
        )

        self.emb_num_lit_img = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim + self.n_num_lit, self.emb_dim),
            torch.nn.Tanh()
        )

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.loss = torch.nn.BCELoss()

    def forward(self, e1, rel):
        # from config
        batch_size = 0

        e1_emb_real = self.emb_e_real(e1).view(batch_size, -1)
        rel_emb_real = self.emb_rel_real(rel).view(batch_size, -1)
        e1_emb_img = self.emb_e_img(e1).view(batch_size, -1)
        rel_emb_img = self.emb_rel_img(rel).view(batch_size, -1)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_emb_real = self.emb_num_lit_real(torch.cat([e1_emb_real, e1_num_lit], 1))
        e1_emb_img = self.emb_num_lit_img(torch.cat([e1_emb_img, e1_num_lit], 1))

        e2_multi_emb_real = self.emb_num_lit_real(torch.cat([self.emb_e_real.weight, self.numerical_literals], 1))
        e2_multi_emb_img = self.emb_num_lit_img(torch.cat([self.emb_e_img.weight, self.numerical_literals], 1))

        # End literals

        e1_emb_real = self.inp_drop(e1_emb_real)
        rel_emb_real = self.inp_drop(rel_emb_real)
        e1_emb_img = self.inp_drop(e1_emb_img)
        rel_emb_img = self.inp_drop(rel_emb_img)

        realrealreal = torch.mm(e1_emb_real * rel_emb_real, e2_multi_emb_real.t())
        realimgimg = torch.mm(e1_emb_real * rel_emb_img, e2_multi_emb_img.t())
        imgrealimg = torch.mm(e1_emb_img * rel_emb_real, e2_multi_emb_img.t())
        imgimgreal = torch.mm(e1_emb_img * rel_emb_img, e2_multi_emb_real.t())

        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred


class ComplexLiterale(KgeModel):

    def __init__(
            self,
            config: Config,
            dataset: Dataset,
            configuration_key=None,
            init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=ComplexLiteraleScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
