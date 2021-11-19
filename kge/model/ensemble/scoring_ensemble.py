from torch import Tensor

from kge import Config, Dataset
from kge.model import Ensemble


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

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        pass

    def compute_scores(self, s:Tensor, p:Tensor, o:Tensor, direction=None) -> Tensor:
        pass

    def combine(self, scores:Tensor):
        pass

    def save(self):
        pass

    def load(self, savepoint):
        pass
    