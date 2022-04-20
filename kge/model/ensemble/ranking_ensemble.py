import torch
from torch import Tensor, tensor

from kge import Config, Dataset
from kge.model import Ensemble
from kge.model.ensemble.ranking_evaluator import AvgRankingEvaluator


class RankingEnsemble(Ensemble):

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
            self.evaluator = AvgRankingEvaluator(config)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        n = s.size()[0]
        res = torch.rand(n)
        return res

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        model_rankings_list = []

        # get all scores from each model
        model_scores = self.model_manager.score_sp(s, p, o)
        num_models = len(self.get_option("base_models"))

        # compute the ranking for each model
        for model_idx in range(num_models):
            scores = model_scores[:, :, model_idx]
            ranking_list = []
            # compute a ranking for each object query
            for row, row_tensor in enumerate(scores):
                n = row_tensor.size()[0]
                ranking_row = torch.zeros(n)
                _, indices = torch.sort(row_tensor, descending=True)
                # assign the rank as negative position in sorted list
                for j, idx in enumerate(indices):
                    ranking_row[idx] = -j
                ranking_list.append(ranking_row)
            model_ranking = torch.stack(ranking_list)
            model_ranking = torch.unsqueeze(model_ranking, dim=-1)
            model_rankings_list.append(model_ranking)

        model_rankings = torch.cat(model_rankings_list, dim=2)
        res = self.evaluator(model_rankings)
        return res

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        model_rankings_list = []

        # get all scores from each model
        model_scores = self.model_manager.score_po(p, o, s)
        num_models = len(self.get_option("base_models"))

        # compute the ranking for each model
        for model_idx in range(num_models):
            scores = model_scores[:, :, model_idx]
            ranking_list = []
            # compute a ranking for each object query
            for row, row_tensor in enumerate(scores):
                n = row_tensor.size()[0]
                ranking_row = torch.zeros(n)
                _, indices = torch.sort(row_tensor, descending=True)
                # assign the rank as negative position in sorted list
                for j, idx in enumerate(indices):
                    ranking_row[idx] = -j
                ranking_list.append(ranking_row)
            model_ranking = torch.stack(ranking_list)
            model_ranking = torch.unsqueeze(model_ranking, dim=-1)
            model_rankings_list.append(model_ranking)

        model_rankings = torch.cat(model_rankings_list, dim=2)
        res = self.evaluator(model_rankings)
        return res

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        model_rankings_list = []

        # get all scores from each model
        model_scores = self.model_manager.score_so(s, o, p)
        num_models = len(self.get_option("base_models"))

        # compute the ranking for each model
        for model_idx in range(num_models):
            scores = model_scores[:, :, model_idx]
            ranking_list = []
            # compute a ranking for each object query
            for row, row_tensor in enumerate(scores):
                n = row_tensor.size()[0]
                ranking_row = torch.zeros(n)
                _, indices = torch.sort(row_tensor, descending=True)
                # assign the rank as negative position in sorted list
                for j, idx in enumerate(indices):
                    ranking_row[idx] = -j
                ranking_list.append(ranking_row)
            model_ranking = torch.stack(ranking_list)
            model_ranking = torch.unsqueeze(model_ranking, dim=-1)
            model_rankings_list.append(model_ranking)

        model_rankings = torch.cat(model_rankings_list, dim=2)
        res = self.evaluator(model_rankings)
        return res

    def score_sp_po(
            self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        sp_scores = self.score_sp(s, p, entity_subset)
        po_scores = self.score_po(entity_subset, p, o)
        return torch.cat((sp_scores, po_scores), dim=1)
