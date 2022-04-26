import torch
from torch import nn, Tensor

from kge import Config, Dataset
from kge.model import KgeModel
from kge.model.kge_model import RelationalScorer


class Gate(nn.Module):
    def __init__(self, input_size, output_size):
        super(Gate, self).__init__()
        self.output_size = output_size

        self.gate_activation = nn.Sigmoid()
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size - output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], 1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1 - gate) * x_ent + gate * g_embedded

        return output


class LiteraleDistmultScorer(RelationalScorer):
    r"""Implementation of the Literale-DistMult KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        if combine == "spo":
            out = (s_emb * p_emb * o_emb).sum(dim=1)
        elif combine == "sp_":
            out = (s_emb * p_emb).mm(o_emb.transpose(0, 1))
        elif combine == "_po":
            out = (o_emb * p_emb).mm(s_emb.transpose(0, 1))
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class LiteraleDistmult(KgeModel):
    r"""Implementation of the LiteralE-DistMult KGE model."""

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
            scorer=LiteraleDistmultScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        entity_dim = self.get_option("entity_embedder.dim")
        num_lit = dataset.get_option("num_numerical_literals")
        self.gru = Gate(entity_dim + num_lit, entity_dim)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        o_emb = self.get_o_embedder().embed(o)

        # enrich embeddings with literal information
        s_literals = self.dataset.load_numerical_literal_emb(s)
        o_literals = self.dataset.load_numerical_literal_emb(o)
        s_emb = self.gru(s_emb, s_literals)
        o_emb = self.gru(o_emb, o_literals)

        return self._scorer.score_emb(s_emb, p_emb, o_emb, combine="spo").view(-1)

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        if o is None:
            o_emb = self.get_o_embedder().embed_all()
        else:
            o_emb = self.get_o_embedder().embed(o)

        # enrich embeddings with literal information
        s_literals = self.dataset.load_numerical_literal_emb(s)
        o_literals = self.dataset.load_numerical_literal_emb(o)
        s_emb = self.gru(s_emb, s_literals)
        o_emb = self.gru(o_emb, o_literals)

        return self._scorer.score_emb(s_emb, p_emb, o_emb, combine="sp_")

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        if s is None:
            s_emb = self.get_s_embedder().embed_all()
        else:
            s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        p_emb = self.get_p_embedder().embed(p)

        # enrich embeddings with literal information
        s_literals = self.dataset.load_numerical_literal_emb(s)
        o_literals = self.dataset.load_numerical_literal_emb(o)
        s_emb = self.gru(s_emb, s_literals)
        o_emb = self.gru(o_emb, o_literals)

        return self._scorer.score_emb(s_emb, p_emb, o_emb, combine="_po")

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        if p is None:
            p_emb = self.get_p_embedder().embed_all()
        else:
            p_emb = self.get_p_embedder().embed(p)

        # enrich embeddings with literal information
        s_literals = self.dataset.load_numerical_literal_emb(s)
        o_literals = self.dataset.load_numerical_literal_emb(o)
        s_emb = self.gru(s_emb, s_literals)
        o_emb = self.gru(o_emb, o_literals)

        return self._scorer.score_emb(s_emb, p_emb, o_emb, combine="s_o")

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        o_emb = self.get_o_embedder().embed(o)

        # enrich embeddings with literal information
        s_literals = self.dataset.load_numerical_literal_emb(s)
        o_literals = self.dataset.load_numerical_literal_emb(o)
        s_emb = self.gru(s_emb, s_literals)
        o_emb = self.gru(o_emb, o_literals)

        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
                all_entities_literals = self.dataset.load_numerical_literal_emb(
                    entity_subset
                )
                all_entities = self.gru(all_entities, all_entities_literals)
            else:
                all_entities = self.get_s_embedder().embed_all()
                all_entities_literals = self.dataset.load_numerical_literal_emb()
                all_entities = self.gru(all_entities, all_entities_literals)
            sp_scores = self._scorer.score_emb(
                s_emb, p_emb, all_entities, combine="sp_"
            )
            po_scores = self._scorer.score_emb(
                all_entities, p_emb, o_emb, combine="_po"
            )
        else:
            if entity_subset is not None:
                all_objects = self.get_o_embedder().embed(entity_subset)
                all_subjects = self.get_s_embedder().embed(entity_subset)
                all_objects_literals = self.dataset.load_numerical_literal_emb(
                    entity_subset
                )
                all_objects = self.gru(all_objects, all_objects_literals)
                all_subjects_literals = self.dataset.load_numerical_literal_emb(
                    entity_subset
                )
                all_subjects = self.gru(all_subjects, all_subjects_literals)
            else:
                all_objects = self.get_o_embedder().embed_all()
                all_subjects = self.get_s_embedder().embed_all()
                all_objects_literals = self.dataset.load_numerical_literal_emb()
                all_objects = self.gru(all_objects, all_objects_literals)
                all_subjects_literals = self.dataset.load_numerical_literal_emb()
                all_subjects = self.gru(all_subjects, all_subjects_literals)
            sp_scores = self._scorer.score_emb(s_emb, p_emb, all_objects, combine="sp_")
            po_scores = self._scorer.score_emb(
                all_subjects, p_emb, o_emb, combine="_po"
            )
        return torch.cat((sp_scores, po_scores), dim=1)
