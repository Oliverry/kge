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
            self.evaluator = PlattScalingEvaluator(config)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        scores = None
        for idx, model in enumerate(self.submodels):
            model_scores = model.score_spo(s, p, o, direction)
            model_scores = torch.unsqueeze(model_scores, dim=-1)
            if scores is None:
                scores = model_scores
            else:
                scores = torch.cat((scores, model_scores), 1)
        return self.evaluator(scores)

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        scores = []
        for idx, model in enumerate(self.submodels):
            model_scores = model.score_sp(s, p)
            model_scores = torch.unsqueeze(model_scores, dim=0)
            model_scores = torch.transpose(model_scores, 0, 1)
            scores.append(model_scores)
        for idx in range(0, 5):
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
        scores = None
        for idx, model in enumerate(self.submodels):
            model_scores = model.score_sp_po(s, p, o, entity_subset)
            model_scores = torch.unsqueeze(model_scores, dim=0)
            if scores is None:
                scores = model_scores
            else:
                scores = torch.cat((scores, model_scores), 0)
        return self.evaluator(scores)

        sp_scores = self.score_sp(s, p)
        po_scores = self.score_po(p, o)
        return torch.cat((sp_scores, po_scores), dim=1)

    def combine(self, scores: Tensor):
        pass

    def save(self):
        pass

    def load(self, savepoint):
        pass
    