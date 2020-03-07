from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from nlpr.tasks.lib.templates.shared import Task, TaskTypes
import nlpr.tasks.lib.mlm as mlm_lib
from nlpr.shared.model_setup import ModelWrapper


def forward_batch_basic(model: nn.Module, batch, omit_label_id: bool = False):
    return model(
        input_ids=batch.input_ids,
        token_type_ids=batch.segment_ids,
        attention_mask=batch.input_mask,
        labels=batch.label_id if not omit_label_id else None,
    )


def forward_batch_delegate(model_wrapper: ModelWrapper,
                           batch,
                           task: Task,
                           omit_label_id: bool = False):
    if task.TASK_TYPE in [TaskTypes.CLASSIFICATION, TaskTypes.REGRESSION]:
        return forward_batch_basic(
            model=model_wrapper.model,
            batch=batch,
            omit_label_id=omit_label_id,
        )[0]
    elif task.TASK_TYPE == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        return model_wrapper.model(
            input_ids=batch.input_ids,
            spans=batch.spans,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_id if not omit_label_id else None,
        )[0]
    elif task.TASK_TYPE == TaskTypes.MULTIPLE_CHOICE:
        return forward_batch_basic(
            model=model_wrapper.model,
            batch=batch,
            omit_label_id=omit_label_id,
        )[0]
    elif task.TASK_TYPE == TaskTypes.SQUAD_STYLE_QA:
        start_positions = batch.start_position if not omit_label_id else None
        end_positions = batch.end_position if not omit_label_id else None
        # TODO: Refactor this wrt model_resolution
        # Actually "xlm", "roberta", "distilbert"
        if model_wrapper.model.__class__.__name__.startswith("Robert"):
            segment_ids = None
        else:
            segment_ids = batch.segment_ids
        logits = model_wrapper.model(
            input_ids=batch.input_ids,
            attention_mask=batch.input_mask,
            token_type_ids=segment_ids,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        return torch.stack(logits, dim=1)
    elif task.TASK_TYPE == TaskTypes.TAGGING:
        return model_wrapper.model(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
        )[0]
    elif task.TASK_TYPE == TaskTypes.MASKED_LANGUAGE_MODELING:
        logits, masked_lm_labels = mlm_forward(
            batch=batch,
            model_wrapper=model_wrapper,
            task=task,
        )
        return MLMOutputTuple(
            logits=logits,
            masked_lm_labels=masked_lm_labels,
            vocab_size=model_wrapper.model.config.vocab_size,
        )
    else:
        raise KeyError(task)


def compute_loss_from_model_output(logits, loss_criterion, batch, task_type: TaskTypes):
    # todo: cleanup
    if task_type == TaskTypes.CLASSIFICATION:
        loss = loss_criterion(logits, batch.label_id)
    elif task_type == TaskTypes.REGRESSION:
        loss = loss_criterion(logits.squeeze(-1), batch.label)
    elif task_type == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        loss = loss_criterion(logits, batch.label_id)
    elif task_type == TaskTypes.MULTIPLE_CHOICE:
        loss = loss_criterion(logits, batch.label_id)
    elif task_type == TaskTypes.SQUAD_STYLE_QA:
        # TODO: Note: we're ignoring the loss_criterion
        loss = qa_compute_loss(
            logits=logits,
            start_positions=batch.start_position,
            end_positions=batch.end_position,
        )
    elif task_type == TaskTypes.TAGGING:
        num_classes = logits.shape[-1]
        if batch.label_mask is not None:
            bool_mask = batch.label_mask.view(-1).bool()
            flat_logits = logits.view(-1, num_classes)[bool_mask]
            flat_labels = batch.label_ids.reshape(-1)[bool_mask]
        else:
            flat_logits = logits.view(-1, num_classes)
            flat_labels = batch.label_ids.view(-1)
        loss = loss_criterion(flat_logits, flat_labels)
    elif task_type == TaskTypes.MASKED_LANGUAGE_MODELING:
        # TODO: THIS IS A HACK
        mlm_output = logits
        loss = loss_criterion(
            mlm_output.logits.view(-1, mlm_output.vocab_size),
            mlm_output.masked_lm_labels.view(-1),
        )
    else:
        raise KeyError(task_type)
    return loss


def mlm_forward(batch, model_wrapper: ModelWrapper, task):

    return logits


def mlm_compute_loss(logits, )


def qa_compute_loss(logits, start_positions, end_positions):
    # Taken from: RobertaForQuestionAnswering

    start_logits, end_logits = logits[:, 0], logits[:, 1]
    # If we are on multi-GPU, split add a dimension
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2
    return total_loss
