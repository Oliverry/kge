import os
from os.path import exists

from kge import Config, Dataset
from kge.job import Job
from kge.misc import pretrained_model_dir
from kge.model.kge_model import KgeModel
from kge.util import load_checkpoint


def load_pretrained_model(config, dataset, model_name) -> KgeModel:
    pretrained_model_path = os.path.join(pretrained_model_dir(), config.get("dataset.name"), model_name)
    pretrained_model_checkpoint_path = os.path.join(pretrained_model_path, "checkpoint_best.pt")
    if exists(pretrained_model_checkpoint_path):
        checkpoint = load_checkpoint(pretrained_model_checkpoint_path, config.get("job.device"))
        pretrained_model_config = Config.create_from(checkpoint)
        pretrained_model_config.set("job.device", config.get("job.device"))
        pretrained_model_config.folder = pretrained_model_path
        model = KgeModel.create(pretrained_model_config, dataset, init_for_load_only=True)
        model.load(checkpoint["model"])
        model.eval()
        return model
    else:
        raise Exception("Could not find pretrained model.")


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
            scorer=None,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        self.submodels = []
        for model in self.get_option("submodels"):
            self.submodels.append(load_pretrained_model(config, dataset, model))

    def prepare_job(self, job: Job, **kwargs):
        super().prepare_job(job, **kwargs)

        from kge.job import TrainingJobNegativeSampling

        if (
                isinstance(job, TrainingJobNegativeSampling)
                and job.config.get("negative_sampling.implementation") == "auto"
        ):
            # TransE with batch currently tends to run out of memory, so we use triple.
            job.config.set("negative_sampling.implementation", "triple", log=True)
