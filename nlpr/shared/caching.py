import math
import numpy as np
import os
from dataclasses import dataclass
from typing import List

import torch
import torch.utils.data

import nlpr.shared.runner as shared_runner

import pyutils.io as io


class Chunker:
    def __init__(self, length, num_chunks, chunk_size):
        self.length = length
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size

    def get_slices(self):
        indices = list(range(0, self.length, self.chunk_size)) + [self.length]
        return [
            slice(start, end)
            for start, end in zip(indices[:-1], indices[1:])
        ]

    def lookup_chunk_and_index(self, i):
        if isinstance(i, int):
            return i // self.chunk_size, i % self.chunk_size
        elif isinstance(i, np.ndarray):
            i = i.astype(int)
            return i / self.chunk_size, i % self.chunk_size
        elif isinstance(i, torch.Tensor):
            return self.lookup_chunk_and_index(i.numpy())
        else:
            raise TypeError(type(i))

    def lookup_index(self, chunk_i, i):
        if isinstance(i, (int, np.ndarray, torch.Tensor)):
            return chunk_i * self.chunk_size + i
        else:
            raise TypeError(type(i))

    @classmethod
    def from_chunk_size(cls, length, chunk_size):
        num_chunks = math.ceil(length / chunk_size)
        return cls(length=length, num_chunks=num_chunks, chunk_size=chunk_size)


@dataclass
class ChunkedData:
    shared_metadata: dict
    chunks: List[dict]

    @property
    def num_chunks(self):
        return len(self.chunks)


def convert_to_chunks(dataset_with_metadata: shared_runner.DatasetWithMetadata, chunk_size: int):
    chunker = Chunker.from_chunk_size(len(dataset_with_metadata.dataset), chunk_size=chunk_size)
    slices = chunker.get_slices()
    shared_metadata = {
        "descriptors": dataset_with_metadata.metadata["descriptors"],
        "other": {},
    }
    chunks = [dict() for _ in range(chunker.num_chunks)]
    for name, descriptor in dataset_with_metadata.get_descriptor_dict().items():
        if descriptor.category in ("dataset", "example_id"):
            for i, data_slice in enumerate(slices):
                chunks[i][name] = dataset_with_metadata.dataset.tensors[descriptor.pos][data_slice]
        elif descriptor.category in ("other_metadata", "other_data"):
            for i, data_slice in enumerate(slices):
                chunks[i][name] = dataset_with_metadata.metadata["other"][name][data_slice]
        else:
            raise KeyError(descriptor.category)
    chunked_data = ChunkedData(shared_metadata=shared_metadata, chunks=chunks)
    assert chunked_data.num_chunks == chunker.num_chunks
    return chunked_data


def chunk_and_save(dataset_with_metadata: shared_runner.DatasetWithMetadata,
                   chunk_size: int,
                   data_args: dict,
                   output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    chunked_data = convert_to_chunks(
        dataset_with_metadata=dataset_with_metadata,
        chunk_size=chunk_size,
    )
    torch.save(chunked_data.shared_metadata, os.path.join(output_dir, "shared_metadata.p"))
    for i, chunk in enumerate(chunked_data.chunks):
        torch.save(chunk, os.path.join(output_dir, f"data_{i:05d}.chunk"))
    data_args = data_args.copy()
    data_args["num_chunks"] = chunked_data.num_chunks
    data_args["length"] = len(dataset_with_metadata.dataset)
    torch.save(data_args, os.path.join(output_dir, "data_args.p"))


def compare_tensor_tuples(tup1, tup2):
    if len(tup1) != len(tup2):
        return False
    for col1, col2 in zip(tup1, tup2):
        if not torch.equal(col1, col2):
            return False
    return True


def compare_dataset_with_metadata(d1, d2):
    if not compare_tensor_tuples(d1.dataset.tensors, d2.dataset.tensors):
        return False
    if not d1.metadata == d2.metadata:
        return False
    return True


class DataCache:
    """
    We're going to liberally use pickling/torch.save/load.
    There is no expectation that caches should be backward compatible.
    """

    def get_all(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class InMemoryDataCache(DataCache):
    def __init__(self, dataset_with_metadata):
        self.dataset_with_metadata = dataset_with_metadata

    def get_all(self):
        return self.dataset_with_metadata

    def __len__(self):
        return len(self.dataset_with_metadata)

    @classmethod
    def from_file(cls, path):
        return torch.load(path)


class ChunkedFilesDataCache(DataCache):
    def __init__(self, cache_fol_path):
        self.cache_fol_path = cache_fol_path
        self.data_args = torch.load(os.path.join(cache_fol_path, "data_args.p"))
        self.num_chunks = self.data_args["num_chunks"]
        self.length = self.data_args["length"]
        self.chunk_size = self.data_args["chunk_size"]
        self.shared_metadata = None
        self.chunker = Chunker.from_chunk_size(length=self.length, chunk_size=self.chunk_size)

    def get_iterable_dataset(self, buffer_size=None):
        if buffer_size is None:
            buffer_size = self.length
        pass

    def get_all(self):
        chunk_ls = []
        for i in range(self.num_chunks):
            chunk_ls.append(torch.load(os.path.join(self.cache_fol_path, f"data_{i:05d}.chunk")))
        if self.shared_metadata is None:
            self.shared_metadata = torch.load(os.path.join(self.cache_fol_path, "shared_metadata.p"))

        max_pos = max(
            descriptor.pos
            for descriptor in self.shared_metadata["descriptors"]
            if descriptor.pos is not None
        )
        dataset = [list() for _ in range(max_pos + 1)]
        metadata = {"descriptors": self.shared_metadata["descriptors"], "other": {}}
        for descriptor in self.shared_metadata["descriptors"]:
            if descriptor.category == "dataset":
                for chunk in chunk_ls:
                    dataset[descriptor.pos].append(chunk[descriptor.name])
            elif descriptor.category in ("other_metadata", "other_data"):
                metadata["other"][descriptor.name] = []
                for chunk in chunk_ls:
                    metadata["other"][descriptor.name] += chunk[descriptor.name]
            else:
                raise KeyError(descriptor.category)
        dataset = [torch.cat(column, dim=0) for column in dataset]
        new_dataset_with_metadata = shared_runner.DatasetWithMetadata(
            dataset=torch.utils.data.TensorDataset(*dataset),
            metadata=metadata,
        )
        return new_dataset_with_metadata

    def __len__(self):
        return self.length
