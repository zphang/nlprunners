import os
import numpy as np

from dataclasses import dataclass
from typing import Dict, Union, NamedTuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from pyutils.display import maybe_tqdm
import pyutils.io as io

from nlpr.tasks.core import BatchMixin
from nlpr.shared.pycore import ExtendedDataClassMixin
from nlpr.shared.model_setup import OptimizerScheduler
from nlpr.shared.modeling.models import forward_batch_delegate, compute_loss_from_model_output
import nlpr.tasks.evaluate as evaluate


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


@dataclass
class DatasetWithMetadata:
    dataset: TensorDataset
    metadata: Dict

    def get_descriptor_dict(self):
        return {
            descriptor.name: descriptor
            for descriptor in self.metadata["descriptors"]
        }


@dataclass
class DataDescriptor:
    category: str
    # category=dataset: Tensor
    # category=other: metadata to carry around be batch
    # category=none: don't propagate
    name: str
    pos: Union[int, None]
    # Tensor position. Only use if category=dataset


def convert_examples_to_dataset(examples, tokenizer, feat_spec, task, verbose=False):
    data_rows = [
        example.tokenize(tokenizer).featurize(tokenizer, feat_spec)
        for example in maybe_tqdm(examples, desc="Tokenizing", verbose=verbose)
    ]
    full_batch = task.Batch.from_data_rows(data_rows)
    dataset_with_metadata = full_batch_to_dataset(full_batch)
    return dataset_with_metadata


def full_batch_to_dataset(full_batch):
    """
    See: HybridLoader for details
    """
    dataset_ls = []
    others_dict = {}
    descriptors = []
    for i, (k, v) in enumerate(full_batch.asdict().items()):
        if isinstance(v, torch.Tensor):
            descriptors.append(DataDescriptor("dataset", k, len(dataset_ls)))
            dataset_ls.append(v)
        elif v is None:
            # Is this even used?
            descriptors.append(DataDescriptor("none", k, None))
        else:
            descriptors.append(DataDescriptor("other_data", k, None))
            others_dict[k] = v

    # We always add example_id as an additional tensor column
    descriptors.append(DataDescriptor(
        "example_id", "example_id", len(dataset_ls)))
    dataset_ls.append(torch.arange(len(full_batch)))
    return DatasetWithMetadata(
        dataset=TensorDataset(*dataset_ls),
        metadata={
            "descriptors": descriptors,
            "other": others_dict,
        }
    )


class BatchTuple(NamedTuple):
    batch: BatchMixin
    metadata: dict

    def to(self, device):
        return BatchTuple(
            batch=self.batch.to(device),
            metadata=self.metadata,
        )


class HybridLoader:
    def __init__(self, dataloader, metadata, task):
        self.dataloader = dataloader
        self.metadata = metadata
        self.task = task

    @property
    def dataset_with_metadata(self):
        return DatasetWithMetadata(
            dataset=self.dataloader.dataset,
            metadata=self.metadata,
        )

    def __iter__(self):
        # dataset: tensors
        # other_data: non-tensors, but go into batch
        # other_metadata: non-tensors, and go into metadata
        descriptor_dict = self.dataset_with_metadata.get_descriptor_dict()
        for batch in self.dataloader:
            example_ids = batch[descriptor_dict["example_id"].pos]
            batch_dict = {}
            batch_metadata_dict = {"example_id": example_ids}
            for name, descriptor in descriptor_dict.items():
                if descriptor.category == "example_id":
                    # Special exception for example_id, cause we already pulled it out
                    continue
                elif descriptor.category == "dataset":
                    batch_dict[name] = batch[descriptor.pos]
                elif descriptor.category == "none":
                    batch_dict[name] = None
                elif descriptor.category == "other_data":
                    batch_dict[name] = [
                        self.metadata["other"][name][i]
                        for i in example_ids
                    ]
                elif descriptor.category == "other_metadata":
                    batch_metadata_dict[name] = [
                        self.metadata["other"][name][i]
                        for i in example_ids
                    ]
                else:
                    raise KeyError(descriptor.category)
            yield BatchTuple(
                batch=self.task.Batch(**batch_dict),
                metadata=batch_metadata_dict,
            )

    def __len__(self):
        return len(self.dataloader)


def get_sampler(dataset, local_rank, force_sequential=False):
    if force_sequential:
        return SequentialSampler(dataset)
    if local_rank == -1:
        return RandomSampler(dataset)
    else:
        return DistributedSampler(dataset)


def run_val(val_examples, val_dataloader,
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
                omit_label_ids=True,
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
        "metrics": evaluate.compute_task_metrics(task, all_logits, val_examples),
    }


def get_train_dataloader(train_examples, task,
                         tokenizer, feat_spec, local_rank, train_batch_size, verbose=True):
    dataset_with_metadata = convert_examples_to_dataset(
        examples=train_examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
        task=task,
        verbose=verbose,
    )
    train_sampler = get_sampler(
        dataset=dataset_with_metadata.dataset,
        local_rank=local_rank,
    )
    train_dataloader = DataLoader(
        dataset=dataset_with_metadata.dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
    )
    return HybridLoader(
        dataloader=train_dataloader,
        metadata=dataset_with_metadata.metadata,
        task=task,
    )


def get_eval_dataloader(eval_examples, task,
                        tokenizer, feat_spec, eval_batch_size):
    dataset_with_metadata = convert_examples_to_dataset(
        examples=eval_examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
        task=task,
    )
    eval_sampler = SequentialSampler(dataset_with_metadata.dataset)
    eval_dataloader = DataLoader(
        dataset=dataset_with_metadata.dataset,
        sampler=eval_sampler,
        batch_size=eval_batch_size,
    )
    return HybridLoader(
        dataloader=eval_dataloader,
        metadata=dataset_with_metadata.metadata,
        task=task,
    )


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
        train_global_state.global_step += 1


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
