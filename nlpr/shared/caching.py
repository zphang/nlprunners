import math
import numpy as np
import os

import torch
import torch.utils.data.dataset

import nlpr.shared.runner as shared_runner


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

    def get_chunks(self, data):
        assert len(data) == self.length
        chunked_data = [data[data_slice] for data_slice in self.get_slices()]
        assert len(chunked_data) == self.num_chunks
        return chunked_data

    def lookup_chunk_and_index(self, i):
        if isinstance(i, int):
            return i // self.chunk_size, i % self.chunk_size
        elif isinstance(i, np.ndarray):
            i = i.astype(int)
            return (i / self.chunk_size).astype(int), (i % self.chunk_size).astype(int)
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


def convert_to_chunks(data, chunk_size: int):
    chunker = Chunker.from_chunk_size(len(data), chunk_size=chunk_size)
    chunked_data = chunker.get_chunks(data)
    return chunked_data


def chunk_and_save(data: list,
                   chunk_size: int,
                   data_args: dict,
                   output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    chunked_data = convert_to_chunks(data=data, chunk_size=chunk_size)
    for i, chunk in enumerate(chunked_data):
        torch.save(chunk, os.path.join(output_dir, f"data_{i:05d}.chunk"))
    data_args = data_args.copy()
    data_args["num_chunks"] = len(chunked_data)
    data_args["length"] = len(data)
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
    def __init__(self, data):
        self.data = data

    def get_all(self):
        return self.data

    def __len__(self):
        return len(self.data)


class ChunkedFilesDataCache(DataCache):
    def __init__(self, cache_fol_path, verbose=False):
        self.cache_fol_path = cache_fol_path

        self.data_args = torch.load(os.path.join(cache_fol_path, "data_args.p"))
        self.num_chunks = self.data_args["num_chunks"]
        self.length = self.data_args["length"]
        self.chunk_size = self.data_args["chunk_size"]
        self.chunker = Chunker.from_chunk_size(length=self.length, chunk_size=self.chunk_size)

    def get_iterable_dataset(self, buffer_size=None, batch_size=1, shuffle=False, subset=None, verbose=False):
        if subset is None:
            subset = self.length
        else:
            assert shuffle
        if buffer_size is None:
            buffer_size = min(self.length, subset)
        buffer_size = math.ceil(buffer_size / batch_size) * batch_size

        indices = np.arange(self.length).astype(int)
        if shuffle:
            np.random.shuffle(indices)
        if subset:
            indices = indices[:subset]
        buffer_chunked_indices = convert_to_chunks(indices, chunk_size=buffer_size)
        return ChunkedFilesIterableDataset(
            buffer_chunked_indices=buffer_chunked_indices,
            chunked_file_data_cache=self,
            verbose=verbose,
        )

    def load_chunk(self, i):
        return torch.load(os.path.join(self.cache_fol_path, f"data_{i:05d}.chunk"))

    def load_from_indices(self, indices, verbose=False):
        chunk_arr, chunk_sub_index_arr = self.chunker.lookup_chunk_and_index(indices)
        reverse_index = np.arange(len(indices)).astype(int)
        result = [None] * len(indices)
        for chunk_i in sorted(list(set(chunk_arr))):
            selector = (chunk_arr == chunk_i)
            chunk = self.load_chunk(chunk_i)
            selected_chunk_sub_index_arr = chunk_sub_index_arr[selector]
            selected_reverse_index = reverse_index[selector]
            if verbose:
                print(f"Loading {len(selected_chunk_sub_index_arr)} indices from chunk {chunk_i}")
            for i, j in zip(selected_chunk_sub_index_arr, selected_reverse_index):
                result[j] = chunk[i]
            del chunk
        return result

    def get_all(self):
        data = []
        for i in range(self.num_chunks):
            data += self.load_chunk(i)
        return shared_runner.ListDataset(data)

    def __len__(self):
        return self.length


class ChunkedFilesIterableDataset(torch.utils.data.dataset.IterableDataset):
    def __init__(self, buffer_chunked_indices, chunked_file_data_cache: ChunkedFilesDataCache, verbose=False):
        self.buffer_chunked_indices = buffer_chunked_indices
        self.chunked_file_data_cache = chunked_file_data_cache
        self.verbose = verbose

    def __iter__(self):
        seen = 0
        total = sum(len(x) for x in self.buffer_chunked_indices)
        for buffer_chunked_index in self.buffer_chunked_indices:
            if self.verbose:
                print(f"Loading buffer {seen} - {seen + len(buffer_chunked_index)} out of {total}")
            buffer = self.chunked_file_data_cache.load_from_indices(buffer_chunked_index, verbose=self.verbose)
            for elem in buffer:
                yield elem
            seen += len(buffer_chunked_index)
