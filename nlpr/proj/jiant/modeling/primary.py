from typing import Dict, Union

import torch.nn as nn

import nlpr.tasks as tasks
import nlpr.proj.jiant.modeling.submodels as submodels
from nlpr.proj.jiant.components.outputs import construct_output_from_dict


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

    def forward(self,
                batch: tasks.BatchMixin,
                task: tasks.Task,
                compute_loss: bool = False):

        if isinstance(batch, dict):
            batch = task.Batch.from_dict(batch)
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
        ).to_dict()


def wrap_jiant_forward(jiant_model: Union[JiantStyleModel, nn.DataParallel],
                       batch: tasks.BatchMixin,
                       task: tasks.Task,
                       compute_loss: bool = False):
    """ Handling multi-gpu ugliness """
    assert isinstance(jiant_model, (JiantStyleModel, nn.DataParallel))
    is_multi_gpu = isinstance(jiant_model, nn.DataParallel)
    model_output = construct_output_from_dict(jiant_model(
        batch=batch.to_dict() if is_multi_gpu else batch,
        task=task,
        compute_loss=compute_loss,
    ))
    if is_multi_gpu:
        model_output.loss = model_output.loss.sum()
    return model_output
