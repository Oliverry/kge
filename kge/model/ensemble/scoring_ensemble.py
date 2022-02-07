import torch
from torch import Tensor

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
            self.evaluator = PlattScalingEvaluator(config, self.configuration_key)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        self.score_sp_po(s, p, o)
        scores_list = []
        for model in self.submodels:
            model_scores = model.score_spo(s, p, o, direction).detach()
            scores_list.append(model_scores)
        scores = torch.stack(scores_list, dim=1)
        res = self.evaluator(scores)
        return res

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        scores_list = []
        for model in self.submodels:
            model_scores = model.score_sp(s, p, o).detach()
            scores_list.append(model_scores)
        scores = torch.stack(scores_list, dim=2)
        res = self.evaluator(scores)
        return res

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        scores_list = []
        for model in self.submodels:
            model_scores = model.score_po(p, o, s).detach()
            scores_list.append(model_scores)
        scores = torch.stack(scores_list, dim=2)
        res = self.evaluator(scores)
        return res

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        scores_list = []
        for model in self.submodels:
            model_scores = model.score_so(s, o, p).detach()
            scores_list.append(model_scores)
        scores = torch.stack(scores_list, dim=2)
        res = self.evaluator(scores)
        return res

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        scores_list = []
        for model in self.submodels:
            model_scores = model.score_sp_po(s, p, o, entity_subset).detach()
            scores_list.append(model_scores)
        scores = torch.stack(scores_list, dim=2)
        res = self.evaluator(scores)
        return res
