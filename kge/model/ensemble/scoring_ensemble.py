import torch
from torch import Tensor

from kge import Config, Dataset
from kge.model import Ensemble
from kge.model.ensemble.scoring_evaluator import AvgScoringEvaluator


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
            self.evaluator = AvgScoringEvaluator()

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        scores = torch.empty((len(self.submodels), s.size(dim=0)))
        for idx, model in enumerate(self.submodels):
            scores[idx] = model.score_spo(s, p, o, direction)
        return self.evaluator(scores)

    def compute_scores(self, s:Tensor, p:Tensor, o:Tensor, direction=None) -> Tensor:
        pass

    def combine(self, scores:Tensor):
        pass

    def save(self):
        pass

    def load(self, savepoint):
        pass
    