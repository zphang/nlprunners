import os
import numpy as np

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from nlpr.shared.preprocessing import convert_examples_to_dataset
from pyutils.display import maybe_tqdm
import pyutils.io as io

from nlpr.shared.pycore import ExtendedDataClassMixin
from nlpr.shared.model_setup import OptimizerScheduler
from nlpr.shared.modeling.models import delegate_forward_and_compute_loss, delegate_forward_batch
import nlpr.tasks.evaluate as evaluate
import nlpr.shared.torch_utils as torch_utils
import nlpr.shared.caching as caching
from nlpr.constants import PHASE


class BaseRunner:
    pass


@dataclass
class TrainGlobalState(ExtendedDataClassMixin):
    epoch: int = 0
    epoch_step: int = 0
    global_step: int = 0

    def step(self):
        self.global_step += 1
        self.epoch_step += 1

    def step_epoch(self):
        self.epoch += 1
        self.epoch_step = 0

    def __str__(self):
        return f"TGS({self.epoch} / {self.epoch_step} ({self.global_step}))"


def get_sampler(dataset, local_rank, force_sequential=False):
    if force_sequential:
        return SequentialSampler(dataset)
    if local_rank == -1:
        return RandomSampler(dataset)
    else:
        return DistributedSampler(dataset)


def run_val(val_dataloader,
            val_labels,
            model_wrapper, task, loss_criterion,
            device, local_rank, verbose):
    # Reminder:
    #   val_dataloader contains mostly PyTorch-relevant info
    #   val_labels might contain more details information needed for full evaluation
    if not local_rank == -1:
        return
    model_wrapper.model.eval()
    total_eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    for step, (batch, batch_metadata) in enumerate(
            maybe_tqdm(val_dataloader, desc="Evaluating (Val)", verbose=verbose)):
        batch = batch.to(device)

        with torch.no_grad():
            batch_logits, batch_loss = delegate_forward_and_compute_loss(
                model_wrapper=model_wrapper,
                batch=batch,
                task=task,
                loss_criterion=loss_criterion,

            )
        batch_logits = batch_logits.detach().cpu().numpy()
        batch_loss = batch_loss.mean().item()
        total_eval_loss += batch_loss
        eval_accumulator.update(
            batch_logits=batch_logits,
            batch_loss=batch_loss,
            batch=batch,
        )

        nb_eval_examples += len(batch)
        nb_eval_steps += 1
    eval_loss = total_eval_loss / nb_eval_steps

    return {
        "accumulator": eval_accumulator,
        "loss": eval_loss,
        "metrics": evaluation_scheme.compute_metrics_from_accumulator(
            task=task,
            accumulator=eval_accumulator,
            labels=val_labels,
            tokenizer=model_wrapper.tokenizer,
        ),
    }


def run_test(test_dataloader,
             model_wrapper, task,
             device, verbose=True):
    model_wrapper.model.eval()
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()
    for step, (batch, batch_metadata) in enumerate(
            maybe_tqdm(test_dataloader, desc="Predictions (Test)", verbose=verbose)):
        batch = batch.to(device)
        with torch.no_grad():
            batch_logits = delegate_forward_batch(
                model_wrapper=model_wrapper,
                batch=batch,
                task=task,
            )
        batch_logits = batch_logits.detach().cpu().numpy()
        eval_accumulator.update(
            batch_logits=batch_logits,
            batch_loss=None,
            batch=batch,
        )

    return eval_accumulator.get_accumulated()


def get_train_dataloader(train_examples, task,
                         tokenizer, feat_spec, local_rank, train_batch_size, verbose=True):
    dataset = convert_examples_to_dataset(
        examples=train_examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
        phase=PHASE.TRAIN,
        verbose=verbose,
    )
    train_sampler = get_sampler(
        dataset=dataset,
        local_rank=local_rank,
    )
    train_dataloader = DataLoader(
        dataset=dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=task.collate_fn,
    )
    return train_dataloader


def get_eval_dataloader(eval_examples, task, phase,
                        tokenizer, feat_spec, eval_batch_size):
    dataset = convert_examples_to_dataset(
        examples=eval_examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
        phase=phase,
    )
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset=dataset,
        sampler=eval_sampler,
        batch_size=eval_batch_size,
        collate_fn=task.collate_fn,
    )
    return eval_dataloader


def get_train_dataloader_from_cache(train_cache: caching.ChunkedFilesDataCache,
                                    task,
                                    train_batch_size: int):
    # Todo: Gin-style config
    dataset = train_cache.get_iterable_dataset(
        buffer_size=10000,
        shuffle=True,
    )
    train_dataloader = torch_utils.DataLoaderWithLength(
        dataset=dataset,
        batch_size=train_batch_size,
        collate_fn=task.collate_fn,
    )
    return train_dataloader


def get_eval_dataloader_from_cache(eval_cache: caching.ChunkedFilesDataCache,
                                   task,
                                   eval_batch_size: int,
                                   subset_num=None,
                                   explicit_subset=None):
    dataset = eval_cache.get_iterable_dataset(
        buffer_size=10000,
        shuffle=False,
        subset_num=subset_num,
        explicit_subset=explicit_subset,
    )
    eval_dataloader = torch_utils.DataLoaderWithLength(
        dataset=dataset,
        batch_size=eval_batch_size,
        collate_fn=task.collate_fn,
    )
    return eval_dataloader


def complex_backpropagate(loss, optimizer, model,
                          fp16, n_gpu, gradient_accumulation_steps, max_grad_norm):
    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    if fp16:
        # noinspection PyUnresolvedReferences
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    return loss


def optim_step_grad_accum(optimizer_scheduler: OptimizerScheduler,
                          train_global_state: TrainGlobalState,
                          gradient_accumulation_steps: int):
    if (train_global_state.epoch_step + 1) % gradient_accumulation_steps == 0:
        optimizer_scheduler.step()
        optimizer_scheduler.optimizer.zero_grad()
    train_global_state.step()


def save_model_with_metadata(model: nn.Module, metadata: dict, output_dir: str, file_name="model"):
    torch.save(
        torch_utils.get_model_for_saving(model).state_dict(),
        os.path.join(output_dir, f"{file_name}.p")
    )
    io.write_json(
        metadata,
        os.path.join(output_dir, f"{file_name}.metadata.json")
    )


def compare_steps_max_steps(step, max_steps):
    return (
        max_steps is not None
        and max_steps != -1
        and step >= max_steps
    )
