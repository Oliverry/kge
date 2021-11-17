import os
from os.path import exists

import torch
from kge import Config, Dataset
from kge.job import Job
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.util import load_checkpoint


def load_pretrained_model(config, dataset, model_name):
    pretrained_model_path = os.path.join(
        "../",
        "local",
        "pretraining",
        config.get("dataset.name"),
        model_name
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
        model = KgeModel.create(pretrained_model_config, dataset, init_for_load_only=True)
        model.load(checkpoint["model"])
        return model
    else:
        return None


class EnsembleScorer(RelationalScorer):
    r"""Implementation of the Ensemble KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        transe_model = load_pretrained_model(config, dataset, "transe")
        complex_model = load_pretrained_model(config, dataset, "complex")
        self.models = [transe_model, complex_model]

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        scores = None
        for model in self.models:
            if scores is None:
                scores = model.score_spo(s_emb, p_emb, o_emb, "spo")
            else:
                scores = torch.cat((scores, model.score_spo(s_emb, p_emb, o_emb, "o")), 0)
        return torch.mean(scores, dim=1)


class Ensemble(KgeModel):
    r"""Implementation of the Ensemble KGE model."""

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
