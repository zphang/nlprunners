import os

from dataclasses import dataclass
from typing import Dict, Union, NamedTuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pyutils.display import maybe_tqdm
import pyutils.io as io

from nlpr.tasks.core import BatchMixin
from nlpr.shared.pycore import ExtendedDataClassMixin
from nlpr.shared.model_setup import OptimizerScheduler
from nlpr.tasks.lib.shared import TaskTypes


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
    dataset_with_metadata.metadata["descriptors"].append(
        DataDescriptor("other_metadata", "example_id", None)
    )
    dataset_with_metadata.metadata["other"]["example_id"] = list(range(len(examples)))
    return dataset_with_metadata


def full_batch_to_dataset(full_batch):
    dataset_ls = []
    others_dict = {}
    descriptors = []
    for i, (k, v) in enumerate(full_batch.asdict().items()):
        if isinstance(v, torch.Tensor):
            descriptors.append(DataDescriptor("dataset", k, len(dataset_ls)))
            dataset_ls.append(v)
        elif v is None:
            descriptors.append(DataDescriptor("none", k, None))
        else:
            descriptors.append(DataDescriptor("other_data", k, None))
            others_dict[k] = v
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
        for batch in self.dataloader:
            example_ids = batch[-1]
            batch_dict = {}
            batch_metadata_dict = {"example_id": example_ids}
            for descriptor in self.metadata["descriptors"]:
                if descriptor.category == "dataset":
                    batch_dict[descriptor.name] = batch[descriptor.pos]
                elif descriptor.category == "none":
                    batch_dict[descriptor.name] = None
                elif descriptor.category == "other_data":
                    batch_dict[descriptor.name] = [
                        self.metadata["other"][descriptor.name][i]
                        for i in example_ids
                    ]
                elif descriptor.category == "other_metadata":
                    batch_metadata_dict[descriptor.name] = [
                        self.metadata["other"][descriptor.name][i]
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


def complex_backpropagate(loss, optimizer, model,
                          fp16, n_gpu, gradient_accumulation_steps, max_grad_norm):
    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    if fp16:
        # noinspection PyUnresolvedReferences
        import amp
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
        max_steps != -1
        and step >= max_steps
    )
