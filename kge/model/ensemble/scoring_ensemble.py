import torch
from torch import Tensor, nn

from kge import Config, Dataset
from kge.model import Ensemble
from kge.model.ensemble.scoring_evaluator import AvgScoringEvaluator, PlattScalingEvaluator


class ScoringEnsemble(Ensemble):

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
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        evaluator_str = self.get_option("evaluator")
        if evaluator_str == "avg":
            self.evaluator = AvgScoringEvaluator(config)
        elif evaluator_str == "platt":
            self.evaluator = PlattScalingEvaluator(config)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        scores = None
        for idx, model in enumerate(self.submodels):
            model_scores = model.score_spo(s, p, o, direction).detach()
            model_scores = torch.unsqueeze(model_scores, dim=-1)
            if scores is None:
                scores = model_scores
            else:
                scores = torch.cat((scores, model_scores), 1)
        res = self.evaluator(scores)
        return res

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        scores = []
        for idx, model in enumerate(self.submodels):
            model_scores = model.score_sp(s, p)
            model_scores = torch.unsqueeze(model_scores, dim=0)
            model_scores = torch.transpose(model_scores, 0, 1)
            scores.append(model_scores)
        for idx in range(0, len(self.submodels)):
            pass
        return self.evaluator(scores)

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        scores = None
        for idx, model in enumerate(self.submodels):
            model_scores = model.score_po(p, o)
            model_scores = torch.unsqueeze(model_scores, dim=0)
            if scores is None:
                scores = model_scores
            else:
                scores = torch.cat((scores, model_scores), 0)
        return self.evaluator(scores)

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        # TODO consider entity subset being none
        col_size = 2*entity_subset.size()[0]
        col_scores = [torch.empty(len(self.submodels), s.size()[0]) for _ in range(0, col_size)]
        res = torch.empty(col_size, s.size()[0])
        for idx, model in enumerate(self.submodels):
            model_scores = model.score_sp_po(s, p, o, entity_subset).detach()
            model_scores = torch.transpose(model_scores, 0, 1)
            for col_idx in range(0, col_size):
                col_scores[col_idx][idx] = model_scores[col_idx]
        for idx in range(0, col_size):
            t = torch.transpose(col_scores[idx], 0, 1)
            col_res = self.evaluator(t)
            res[idx] = col_res
        res = torch.transpose(res, 0, 1)
        return res

    