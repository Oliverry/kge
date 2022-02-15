import os
from os.path import exists
from typing import List

from torch import Tensor

from kge import Config, Dataset
from kge.job import Job
from kge.misc import pretrained_model_dir
from kge.model.ensemble.model_manager import ModelManager
from kge.model.kge_model import KgeModel, KgeEmbedder, RelationalScorer, KgeBase
from kge.util import load_checkpoint


class Ensemble(KgeModel):
    """
    Implementation of the Ensemble KGE model.
    Creates no embedders.
    Loads base model list used for the ensemble.
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
            scorer=None,
            create_embedders=False,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only
        )
        base_models_names = self.get_option("base_models")
        base_models = []
        for model_name in base_models_names:
            model = self.load_pretrained_model(model_name)
            # check if all models use a single embedder for subjects and objects
            if not model.get_s_embedder() is model.get_o_embedder():
                raise Exception(
                    "Ensemble only support KGE models with the same s and o embedder. Exception: " + model_name
                )
            base_models.append(model)
        self.model_manager = ModelManager(base_models)

    def load_pretrained_model(self, model_name) -> KgeModel:
        """
        The pretrained model has to be saved in the folder dataset/modelname
        :param model_name:
        :return:
        """
        pretrained_model_path = os.path.join(pretrained_model_dir(), self.config.get("dataset.name"), model_name)
        pretrained_model_checkpoint_path = os.path.join(pretrained_model_path, "checkpoint_best.pt")
        if exists(pretrained_model_checkpoint_path):
            checkpoint = load_checkpoint(pretrained_model_checkpoint_path, self.config.get("job.device"))
            pretrained_model_config = Config.create_from(checkpoint)
            pretrained_model_config.set("job.device", self.config.get("job.device"))
            pretrained_model_config.folder = pretrained_model_path
            # TODO use create_from?
            model = KgeModel.create(pretrained_model_config, self.dataset, init_for_load_only=True)
            model.load(checkpoint["model"])
            model.eval()
            return model
        else:
            raise Exception("Could not find pretrained model.")

    def prepare_job(self, job: "Job", **kwargs):
        from kge.job import TrainingOrEvaluationJob

        if isinstance(job, TrainingOrEvaluationJob):

            def append_num_parameter(job):
                job.current_trace["epoch"]["num_parameters"] = sum(
                    map(lambda p: p.numel(), job.model.parameters())
                )

            job.post_epoch_hooks.append(append_num_parameter)

    def penalty(self, **kwargs) -> List[Tensor]:
        """
        Add regularization penalty of the ensemble model.
        :param kwargs:
        :return:
        """
        p = KgeBase.penalty(self, **kwargs)
        return p

    def get_s_embedder(self) -> KgeEmbedder:
        raise Exception("The ensemble model does not support an own subject embedder.")

    def get_o_embedder(self) -> KgeEmbedder:
        raise Exception("The ensemble model does not support on own object embedder.")

    def get_p_embedder(self) -> KgeEmbedder:
        raise Exception("The ensemble model does not support an own predicate embedder.")

    def get_scorer(self) -> RelationalScorer:
        raise Exception("The ensemble model does not support an own relational scorer.")
