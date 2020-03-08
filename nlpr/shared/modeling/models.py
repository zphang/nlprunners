import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from nlpr.tasks.lib.templates.shared import Task, TaskTypes
import nlpr.tasks.lib.mlm as mlm_lib
from nlpr.shared.model_setup import ModelWrapper


def forward_batch_basic(model: nn.Module, batch):
    return model(
        input_ids=batch.input_ids,
        token_type_ids=batch.segment_ids,
        attention_mask=batch.input_mask,
    )


def delegate_forward_batch(model_wrapper: ModelWrapper,
                           batch,
                           task: Task):
    if task.TASK_TYPE in [TaskTypes.CLASSIFICATION, TaskTypes.REGRESSION]:
        return forward_batch_basic(
            model=model_wrapper.model,
            batch=batch,
        )[0]
    elif task.TASK_TYPE == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        return model_wrapper.model(
            input_ids=batch.input_ids,
            spans=batch.spans,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
        )[0]
    elif task.TASK_TYPE == TaskTypes.MULTIPLE_CHOICE:
        return forward_batch_basic(
            model=model_wrapper.model,
            batch=batch,
        )[0]
    elif task.TASK_TYPE == TaskTypes.SQUAD_STYLE_QA:
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
            start_positions=batch.start_positions,
            end_positions=batch.end_positions,
        )
        return torch.stack(logits, dim=1)
    elif task.TASK_TYPE == TaskTypes.TAGGING:
        return model_wrapper.model(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
        )[0]
    elif task.TASK_TYPE == TaskTypes.MASKED_LANGUAGE_MODELING:
        logits, _ = mlm_forward(
            batch=batch,
            model_wrapper=model_wrapper,
            task=task,
        )
        return logits
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
        assert isinstance(batch, mlm_lib.MaskedBatch), "Can only directly compute loss from MaskedBatch"
        mlm_output = logits
        loss = loss_criterion(
            mlm_output.logits.view(-1, mlm_output.vocab_size),
            mlm_output.masked_lm_labels.view(-1),
        )
    else:
        raise KeyError(task_type)
    return loss


def delegate_forward_and_compute_loss(model_wrapper: ModelWrapper,
                                      batch,
                                      task: Task,
                                      loss_criterion):
    if task.TASK_TYPE in [
            TaskTypes.CLASSIFICATION,
            TaskTypes.REGRESSION,
            TaskTypes.SPAN_COMPARISON_CLASSIFICATION,
            TaskTypes.MULTIPLE_CHOICE,
            TaskTypes.SQUAD_STYLE_QA,
            TaskTypes.TAGGING]:
        logits = delegate_forward_batch(
            model_wrapper=model_wrapper,
            batch=batch,
            task=task,
        )
        loss = compute_loss_from_model_output(
            logits=logits,
            batch=batch,
            loss_criterion=loss_criterion,
            task_type=task.TASK_TYPE,
        )
    elif task.TASK_TYPE == TaskTypes.MASKED_LANGUAGE_MODELING:
        assert (
            isinstance(batch, mlm_lib.MaskedBatch),
            "Can only directly compute loss from MaskedBatch"
        )
        logits, masked_batch = mlm_forward(
            batch=batch,
            model_wrapper=model_wrapper,
            task=task,
        )
        loss = loss_criterion(
            logits.view(-1, model_wrapper.model.config.vocab_size),
            masked_batch.masked_lm_labels.view(-1),
        )
    else:
        raise TypeError(type(task))
    return logits, loss


def mlm_forward(batch, model_wrapper: ModelWrapper, task):
    if isinstance(batch, mlm_lib.Batch):
        masked_batch = batch.get_masked(
            mlm_probability=task.mlm_probability,
            tokenizer=model_wrapper.tokenizer,
        )
    elif isinstance(batch, mlm_lib.MaskedBatch):
        masked_batch = batch
    else:
        raise TypeError(type(batch))
    logits = model_wrapper.model(
        input_ids=masked_batch.masked_input_ids,
        token_type_ids=masked_batch.segment_ids,
        attention_mask=masked_batch.input_mask,
        # masked_lm_labels=masked_batch.masked_lm_labels,
    )[0]
    return logits, masked_batch


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
