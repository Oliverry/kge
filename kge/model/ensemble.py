import os
from os.path import exists

import torch
from kge import Config, Dataset
from kge.job import Job
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as F

from kge.util import load_checkpoint
from kge.util.io import get_checkpoint_file


class EnsembleScorer(RelationalScorer):
    r"""Implementation of the TransE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

        # pretrained model checkpoint
        pretrained_model_path = os.path.join(
            "../",
            "local",
            "pretraining",
            "fb15k-237",
            "transe",
        )
        pretrained_model_config = Config()
        pretrained_model_config.set("job.device", config.get("job.device"))
        pretrained_model_config_path = os.path.join(
            pretrained_model_path,
            "config.yaml"
        )
        if exists(pretrained_model_config_path):
            pretrained_model_config.load(pretrained_model_config_path)
            pretrained_model_config.folder = pretrained_model_path
            pretrained_model_checkpoint_file = pretrained_model_config.checkpoint_file("best")
            checkpoint = load_checkpoint(pretrained_model_checkpoint_file, pretrained_model_config.get("job.device"))
            self.model = KgeModel.create(pretrained_model_config, dataset, init_for_load_only=True)
            self.model.load(checkpoint["model"])

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        score = self.model.score_spo(s_emb, p_emb, o_emb, "o")
        return score


class Ensemble(KgeModel):
    r"""Implementation of the TransE KGE model."""

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
            scorer=EnsembleScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )

    def prepare_job(self, job: Job, **kwargs):
        super().prepare_job(job, **kwargs)

        from kge.job import TrainingJobNegativeSampling

        if (
                isinstance(job, TrainingJobNegativeSampling)
                and job.config.get("negative_sampling.implementation") == "auto"
        ):
            # TransE with batch currently tends to run out of memory, so we use triple.
            job.config.set("negative_sampling.implementation", "triple", log=True)
