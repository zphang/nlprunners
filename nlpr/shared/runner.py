import os
import numpy as np

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from pyutils.display import maybe_tqdm
import pyutils.io as io

from nlpr.shared.pycore import ExtendedDataClassMixin
from nlpr.shared.model_setup import OptimizerScheduler
from nlpr.shared.modeling.models import forward_batch_delegate, compute_loss_from_model_output
import nlpr.tasks.evaluate as evaluate
import nlpr.shared.torch_utils as torch_utils
import nlpr.shared.caching as caching


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


def convert_examples_to_dataset(examples, tokenizer, feat_spec, verbose=False):
    data_rows = [
        example.tokenize(tokenizer).featurize(tokenizer, feat_spec)
        for example in maybe_tqdm(examples, desc="Tokenizing", verbose=verbose)
    ]
    metadata = {
        "example_id": list(range(len(data_rows))),
    }
    data = []
    for i, data_row in enumerate(data_rows):
        metadata_row = {
            k: v[i]
            for k, v in metadata.items()
        }
        data.append({"data_row": data_row, "metadata": metadata_row})
    return torch_utils.ListDataset(data)


def get_sampler(dataset, local_rank, force_sequential=False):
    if force_sequential:
        return SequentialSampler(dataset)
    if local_rank == -1:
        return RandomSampler(dataset)
    else:
        return DistributedSampler(dataset)


def run_val(val_dataloader,
            model, task, loss_criterion,
            device, local_rank, verbose):
    if not local_rank == -1:
        return
    model.eval()
    total_eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_logits = []
    for step, (batch, batch_metadata) in enumerate(
            maybe_tqdm(val_dataloader, desc="Evaluating (Val)", verbose=verbose)):
        batch = batch.to(device)

        with torch.no_grad():
            logits = forward_batch_delegate(
                model=model,
                batch=batch,
                omit_label_id=True,
                task_type=task.TASK_TYPE,
            )[0]
            tmp_eval_loss = compute_loss_from_model_output(
                logits=logits,
                loss_criterion=loss_criterion,
                batch=batch,
                task_type=task.TASK_TYPE,
            )

        logits = logits.detach().cpu().numpy()
        total_eval_loss += tmp_eval_loss.mean().item()

        nb_eval_examples += len(batch)
        nb_eval_steps += 1
        all_logits.append(logits)
    eval_loss = total_eval_loss / nb_eval_steps
    all_logits = np.concatenate(all_logits, axis=0)

    return {
        "logits": all_logits,
        "loss": eval_loss,
        "metrics": evaluate.compute_task_metrics(task, all_logits,
                                                 task.get_val_examples()),
    }


def get_train_dataloader(train_examples, task,
                         tokenizer, feat_spec, local_rank, train_batch_size, verbose=True):
    dataset = convert_examples_to_dataset(
        examples=train_examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
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


def get_eval_dataloader(eval_examples, task,
                        tokenizer, feat_spec, eval_batch_size):
    dataset = convert_examples_to_dataset(
        examples=eval_examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
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
                                   subset=None):
    dataset = eval_cache.get_iterable_dataset(
        buffer_size=10000,
        shuffle=False,
        subset=subset,
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
        model.state_dict(),
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
