import numpy as np
import torch

from nlpr.shared import torch_utils as torch_utils
from nlpr.tasks.core import FeaturizationSpec
from nlpr.tasks import SquadTask, MultiQATask
import nlpr.shared.caching as shared_caching

from pyutils.display import maybe_tqdm


class MaxValidLengthRecorder:
    def __init__(self, max_seq_length):
        self.max_valid_length = 0
        self.max_seq_length = max_seq_length
        self.range_idx = np.arange(max_seq_length)

    def __call__(self, datum):
        if "input_mask" not in datum["data_row"].get_fields():
            raise RuntimeError("Smart truncate not supported")
        indexer = datum["data_row"].input_mask.reshape(-1, self.max_seq_length).max(-2)
        valid_length = self.range_idx[indexer.astype(bool)].max() + 1
        self.max_valid_length = max(self.max_valid_length, valid_length)


def experimental_smart_truncate(dataset: torch_utils.ListDataset,
                                max_seq_length: int):
    if "input_mask" not in dataset.data[0]["data_row"].get_fields():
        raise RuntimeError("Smart truncate not supported")
    valid_length_ls = []
    range_idx = np.arange(max_seq_length)
    for datum in dataset.data:
        indexer = datum["data_row"].input_mask.reshape(-1, max_seq_length).max(-2)
        valid_length_ls.append(range_idx[indexer.astype(bool)].max() + 1)
    max_valid_length = max(valid_length_ls)

    if max_valid_length == max_seq_length:
        return dataset, max_seq_length

    new_datum_ls = []
    for datum in dataset.data:
        new_datum_ls.append(experimental_smart_truncate_datum(
            datum=datum,
            max_seq_length=max_seq_length,
            max_valid_length=max_valid_length,
        ))
    new_dataset = torch_utils.ListDataset(new_datum_ls)
    return new_dataset, max_valid_length


def experimental_smart_truncate_cache(cache: shared_caching.ChunkedFilesDataCache,
                                      max_seq_length: int,
                                      max_valid_length: int):
    for chunk_i in range(cache.num_chunks):
        chunk = torch.load(cache.get_chunk_path(chunk_i))
        new_chunk = [
            experimental_smart_truncate_datum(
                datum=datum,
                max_seq_length=max_seq_length,
                max_valid_length=max_valid_length,
            )
            for datum in chunk
        ]
        torch.save(new_chunk, cache.get_chunk_path(chunk_i))


def experimental_smart_truncate_datum(datum, max_seq_length, max_valid_length):
    row_dict = datum["data_row"].asdict()
    new_row_dict = row_dict.copy()
    for k, v in row_dict.items():
        if not isinstance(v, np.ndarray):
            continue
        if max_seq_length not in v.shape:
            continue
        if not v.shape.count(max_seq_length) == 1:
            raise RuntimeError("confusing dimensions")
        slice_ls = []
        for n in v.shape:
            if n == max_seq_length:
                slice_ls.append(slice(None, max_valid_length))
            else:
                slice_ls.append(slice(None))
        new_row_dict[k] = v[tuple(slice_ls)]
    return {
        "data_row": datum["data_row"].__class__(**new_row_dict),
        "metadata": datum["metadata"],
    }


def convert_examples_to_dataset(examples: list,
                                tokenizer,
                                feat_spec: FeaturizationSpec,
                                phase: str,
                                verbose=False):
    data_rows = tokenize_and_featurize(
        examples=examples,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
        phase=phase,
        verbose=verbose,
    )
    metadata = {"example_id": list(range(len(data_rows)))}
    data = []
    for i, data_row in enumerate(data_rows):
        metadata_row = {
            k: v[i]
            for k, v in metadata.items()
        }
        data.append({"data_row": data_row, "metadata": metadata_row})
    return torch_utils.ListDataset(data)


def iter_chunk_convert_examples_to_dataset(examples: list,
                                           tokenizer,
                                           feat_spec: FeaturizationSpec,
                                           phase: str,
                                           verbose=False):
    for i, data_row in enumerate(iter_chunk_tokenize_and_featurize(
                examples=examples,
                tokenizer=tokenizer,
                feat_spec=feat_spec,
                phase=phase,
                verbose=verbose,
            )):
        metadata = {"example_id": i}
        yield {"data_row": data_row, "metadata": metadata}


def tokenize_and_featurize(examples: list,
                           tokenizer,
                           feat_spec: FeaturizationSpec,
                           phase,
                           verbose=False):
    # TODO: Better solution
    if isinstance(examples[0], (SquadTask.Example, MultiQATask.Example)):
        data_rows = []
        for example in maybe_tqdm(examples, desc="Tokenizing", verbose=verbose):
            # TODO more arguments?
            data_rows += example.to_feature_list(
                tokenizer=tokenizer,
                feat_spec=feat_spec,
                max_seq_length=feat_spec.max_seq_length,
                doc_stride=128,
                max_query_length=64,
                set_type=phase,
            )
    else:
        data_rows = [
            example.tokenize(tokenizer).featurize(tokenizer, feat_spec)
            for example in maybe_tqdm(examples, desc="Tokenizing", verbose=verbose)
        ]
    return data_rows


def iter_chunk_tokenize_and_featurize(examples: list,
                                      tokenizer,
                                      feat_spec: FeaturizationSpec,
                                      phase,
                                      verbose=False):
    # TODO: Better solution
    if isinstance(examples[0], (SquadTask.Example, MultiQATask.Example)):
        for example in maybe_tqdm(examples, desc="Tokenizing", verbose=verbose):
            # TODO more arguments?
            yield from example.to_feature_list(
                tokenizer=tokenizer,
                feat_spec=feat_spec,
                max_seq_length=feat_spec.max_seq_length,
                doc_stride=128,
                max_query_length=64,
                set_type=phase,
            )
    else:
        for example in maybe_tqdm(examples, desc="Tokenizing", verbose=verbose):
            yield example.tokenize(tokenizer).featurize(tokenizer, feat_spec)
