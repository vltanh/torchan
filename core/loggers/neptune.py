import os
from typing import Optional, Dict, List
import neptune
import sys

from .logger_template import LoggerTemplate

class NeptuneLogger(LoggerTemplate):
    def __init__(
        self,
        project_name: str,
        name: str,
        api_token: Optional[str]=None,
        model_params: Optional[Dict]=None,
        tags: Optional[List[str]]=None,
        mode="async"
    ):
        """
        Args:
            project_name: Name of Neptune Project
                ex: my_workspace/my_project
            name: Experiment name
                ex: mnist-resnet18
            api_token: API token to access Neptune account
            model_params: Model parameters for neptune to log down
            tags: List tags for neptune client

            mode: async or offline is two main Neptune mode
        """
        if api_token is None:
            api_token = os.environ.get("NEPTUNE_API_TOKEN", None)
            if api_token is None:
                raise ValueError("NEPTUNE_API_TOKEN not found")

        run = neptune.init_run(
            project=project_name,
            api_token=api_token,
            source_files=["*.py"],
            mode=mode,
        )

        if model_params is not None:
            run["parameters"] = model_params

        if tags is not None:
            run["sys/tags"].add(*tags)

        self.run = run
        self.run["run_cmd"] = ' '.join(sys.argv)


    def update_scalar(self, tag, value, step):
        self.run[tag].log(value, step=step)

    def update_loss(self, phase, value, step):
        tag = f"{phase}/loss"
        self.update_scalar(tag, value, step)

    def update_metric(self, phase, metric, value, step):
        tag = f"{phase}/{metric}"
        self.update_scalar(tag, value, step)

    def update_lr(self, gid, value, step):
        tag = f"lr/group_{gid}"
        self.update_scalar(tag, value, step)
