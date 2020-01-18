import torch
import torch.nn as nn

from nlpr.tasks.lib.shared import TaskTypes


class MultiTaskModel(nn.Module):
    def __init__(self, model_dict, shared_ptt_encoder):
        super().__init__()
        self.model_dict = nn.ModuleDict(model_dict)
        self.shared_ptt_encoder = shared_ptt_encoder

    def forward(self, *args, **kwargs):
        task_name = kwargs.pop("task_name")
        return self.model_dict[task_name].forward(*args, **kwargs)


def forward_batch_basic(model: nn.Module, batch, task_name, omit_label_id: bool = False):
    return model(
        input_ids=batch.input_ids,
        token_type_ids=batch.segment_ids,
        attention_mask=batch.input_mask,
        labels=batch.label_id if not omit_label_id else None,
        task_name=task_name,
    )


def forward_batch_delegate(model: nn.Module, batch,
                           task_name, task_type: TaskTypes, omit_label_id: bool = False):
    # TODO: Why is this copied?
    if task_type in [TaskTypes.CLASSIFICATION, TaskTypes.REGRESSION]:
        return forward_batch_basic(
            model=model,
            batch=batch,
            omit_label_id=omit_label_id,
            task_name=task_name,
        )[0]
    elif task_type == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        spans = torch.stack([
            batch.sentence1_span,
            batch.sentence2_span,
        ], dim=-2)
        return model(
            input_ids=batch.input_ids,
            spans=spans,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_id if not omit_label_id else None,
            task_name=task_name,
        )[0]
    elif task_type == TaskTypes.MULTIPLE_CHOICE:
        return forward_batch_basic(
            model=model,
            batch=batch,
            omit_label_id=omit_label_id,
            task_name=task_name,
        )[0]
    elif task_type == TaskTypes.SQUAD_STYLE_QA:
        if batch.start_position is None:
            start_position = batch.start_position
        else:
            start_position = None
        if batch.end_position is None:
            end_position = batch.end_position
        else:
            end_position = None
        return model(
            input_ids=batch.input_ids,
            attention_mask=batch.input_mask,
            token_type_ids=batch.segment_ids,
            start_position=start_position,
            end_position=end_position,
        )
    else:
        raise KeyError(task_type)
