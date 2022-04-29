from typing import List

from torch import Tensor

from kge import Config, Dataset
from kge.model import Ensemble
from kge.model.ensemble.scoring_evaluator import (
    AvgScoringEvaluator,
    PlattScalingEvaluator,
)


class ScoringEnsemble(Ensemble):
    """
    Ensemble type to combine the score of base models to an overall final score.
    The aggregation of scores is specified by the evaluator.
    """

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

        # create the specified evaluator
        evaluator_str = self.get_option("evaluator")
        if evaluator_str == "avg":
            self.evaluator = AvgScoringEvaluator(config)
            config.log("Created avg scoring evaluator.")
        elif evaluator_str == "platt":
            self.evaluator = PlattScalingEvaluator(config, self.configuration_key)
            config.log("Created platt scoring evaluator.")

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        scores = self.model_manager.score_spo(s, p, o, direction)
        res = self.evaluator(scores)
        return res

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        scores = self.model_manager.score_sp(s, p, o)
        res = self.evaluator(scores)
        return res

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        scores = self.model_manager.score_po(p, o, s)
        res = self.evaluator(scores)
        return res

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        scores = self.model_manager.score_so(s, o, p)
        res = self.evaluator(scores)
        return res

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        scores = self.model_manager.score_sp_po(s, p, o, entity_subset)
        res = self.evaluator(scores)
        return res

    def penalty(self, **kwargs) -> List[Tensor]:
        result = super().penalty(**kwargs)
        result += self.evaluator.penalty(**kwargs)
        return result
