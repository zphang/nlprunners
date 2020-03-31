from typing import Dict

import torch.nn as nn

import nlpr.tasks as tasks
import nlpr.proj.jiant.modeling.submodels as submodels


class JiantStyleModel(nn.Module):
    def __init__(self,
                 task_dict: Dict[str, tasks.Task],
                 encoder: nn.Module,
                 submodels_dict: Dict[str, submodels.Submodel],
                 task_to_submodel_map: Dict[str, str],
                 tokenizer):
        super().__init__()
        self.task_dict = task_dict
        self.encoder = encoder
        self.submodels_dict = nn.ModuleDict(submodels_dict)
        self.task_to_submodel_map = task_to_submodel_map
        self.tokenizer = tokenizer

    def forward(self, batch, task, compute_loss: bool = False):
        if isinstance(task, str):
            task_name = task
            task = self.task_dict[task]
        else:
            task_name = task.name
            task = task
        submodel_key = self.task_to_submodel_map[task_name]
        submodel = self.submodels_dict[submodel_key]
        return submodel(
            batch=batch,
            task=task,
            tokenizer=self.tokenizer,
            compute_loss=compute_loss,
        )
