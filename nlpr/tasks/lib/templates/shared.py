from __future__ import annotations

import csv
import json
import numpy as np
from typing import List, NamedTuple, Mapping
from enum import Enum

import torch
import torch.utils.data.dataloader as dataloader

from dataclasses import dataclass

from nlpr.tasks.core import FeaturizationSpec, BatchMixin
from nlpr.tasks.utils import truncate_sequences, pad_to_max_seq_length

from pyutils.datastructures import combine_dicts


class TaskTypes(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2
    SPAN_COMPARISON_CLASSIFICATION = 3
    MULTIPLE_CHOICE = 4
    SPAN_CHOICE_PROB_TASK = 5
    SQUAD_STYLE_QA = 6
    TAGGING = 7
    MASKED_LANGUAGE_MODELING = 8
    UNDEFINED = -1


class BatchTuple(NamedTuple):
    batch: BatchMixin
    metadata: dict

    def to(self, device):
        return BatchTuple(
            batch=self.batch.to(device),
            metadata=self.metadata,
        )


def metadata_collate_fn(metadata: list):
    return {
        k: [d[k] for d in metadata]
        for k in metadata[0].keys()
    }


def flat_collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, (np.ndarray, int, float, str)):
        return dataloader.default_collate(batch)
    elif isinstance(elem, (list, dict)):
        # Don't do anything to list of lists
        return batch
    else:
        raise TypeError(type(batch))


class Task:

    Example = NotImplemented
    TokenizedExample = NotImplemented
    DataRow = NotImplemented
    Batch = NotImplemented

    TASK_TYPE = NotImplemented

    def __init__(self, name, path_dict):
        self.name = name
        self.path_dict = path_dict

    @property
    def train_path(self):
        return self.path_dict["train"]

    @property
    def val_path(self):
        return self.path_dict["val"]

    @property
    def test_path(self):
        return self.path_dict["test"]

    @classmethod
    def collate_fn(cls, batch: Batch):
        # cls.collate_fn
        elem = batch[0]
        if isinstance(elem, Mapping):  # dict
            assert set(elem.keys()) == {"data_row", "metadata"}
            data_rows = [x["data_row"] for x in batch]
            metadata = [x["metadata"] for x in batch]
            collated_data_rows = {
                key: flat_collate_fn([getattr(d, key) for d in data_rows])
                for key in data_rows[0].asdict()
            }
            collated_metadata = metadata_collate_fn(metadata)
            combined = combine_dicts([collated_data_rows, collated_metadata])
            batch_dict = {}
            for field, field_type in cls.Batch.get_annotations().items():
                batch_dict[field] = combined.pop(field)
                if field_type == torch.FloatTensor:
                    # Ensure that floats stay as float32
                    batch_dict[field] = batch_dict[field].float()
            out_batch = cls.Batch(**batch_dict)
            remainder = combined
            return out_batch, remainder
        else:
            raise TypeError(f"Unknown type for collate_fn {type(elem)}")


class Span(NamedTuple):
    start: int
    end: int  # Use exclusive end, for consistency

    def add(self, i: int) -> Span:
        return Span(start=self.start + i, end=self.end + i)

    def to_slice(self):
        return slice(*self)

    def to_array(self):
        return np.array([self.start, self.end])


@dataclass
class UnpaddedInputs:
    unpadded_tokens: List
    unpadded_segment_ids: List
    cls_offset: int


@dataclass
class InputSet:
    input_ids: List
    input_mask: List
    segment_ids: List


def single_sentence_featurize(guid: str,
                              input_tokens: List[str],
                              label_id: int,
                              tokenizer,
                              feat_spec: FeaturizationSpec,
                              data_row_class):
    unpadded_inputs = construct_single_input_tokens_and_segment_ids(
        input_tokens=input_tokens,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )
    return create_generic_data_row_from_tokens_and_segments(
        guid=guid,
        unpadded_tokens=unpadded_inputs.unpadded_tokens,
        unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
        label_id=label_id,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
        data_row_class=data_row_class,
    )


def double_sentence_featurize(guid: str,
                              input_tokens_a: List[str],
                              input_tokens_b: List[str],
                              label_id: int,
                              tokenizer,
                              feat_spec: FeaturizationSpec,
                              data_row_class):
    unpadded_inputs = construct_double_input_tokens_and_segment_ids(
        input_tokens_a=input_tokens_a,
        input_tokens_b=input_tokens_b,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )

    return create_generic_data_row_from_tokens_and_segments(
        guid=guid,
        unpadded_tokens=unpadded_inputs.unpadded_tokens,
        unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
        label_id=label_id,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
        data_row_class=data_row_class,
    )


def construct_single_input_tokens_and_segment_ids(input_tokens: List[str],
                                                  tokenizer,
                                                  feat_spec: FeaturizationSpec):
    special_tokens_count = 2  # CLS, SEP

    input_tokens, = truncate_sequences(
        tokens_ls=[input_tokens],
        max_length=feat_spec.max_seq_length - special_tokens_count,
    )

    return add_cls_token(
        unpadded_tokens=input_tokens + [tokenizer.sep_token],
        unpadded_segment_ids=(
            [feat_spec.sequence_a_segment_id]
            + [feat_spec.sequence_a_segment_id] * (len(input_tokens))
        ),
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )


def construct_double_input_tokens_and_segment_ids(input_tokens_a: List[str],
                                                  input_tokens_b: List[str],
                                                  tokenizer,
                                                  feat_spec: FeaturizationSpec):

    if feat_spec.sep_token_extra:
        maybe_extra_sep = [tokenizer.sep_token]
        maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
        special_tokens_count = 4  # CLS, SEP-SEP, SEP
    else:
        maybe_extra_sep = []
        maybe_extra_sep_segment_id = []
        special_tokens_count = 3  # CLS, SEP, SEP

    input_tokens_a, input_tokens_b = truncate_sequences(
        tokens_ls=[input_tokens_a, input_tokens_b],
        max_length=feat_spec.max_seq_length - special_tokens_count,
    )

    unpadded_tokens = (
        input_tokens_a + [tokenizer.sep_token]
        + maybe_extra_sep
        + input_tokens_b + [tokenizer.sep_token]
    )
    unpadded_segment_ids = (
        [feat_spec.sequence_a_segment_id] * len(input_tokens_a)
        + [feat_spec.sequence_a_segment_id]
        + maybe_extra_sep_segment_id
        + [feat_spec.sequence_b_segment_id] * len(input_tokens_b)
        + [feat_spec.sequence_b_segment_id]
    )
    return add_cls_token(
        unpadded_tokens=unpadded_tokens,
        unpadded_segment_ids=unpadded_segment_ids,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )


def add_cls_token(unpadded_tokens: List[str],
                  unpadded_segment_ids: List[int],
                  tokenizer,
                  feat_spec: FeaturizationSpec):
    if feat_spec.cls_token_at_end:
        return UnpaddedInputs(
            unpadded_tokens=unpadded_tokens + [tokenizer.cls_token],
            unpadded_segment_ids=unpadded_segment_ids + [feat_spec.cls_token_segment_id],
            cls_offset=0,
        )
    else:
        return UnpaddedInputs(
            unpadded_tokens=[tokenizer.cls_token] + unpadded_tokens,
            unpadded_segment_ids=[feat_spec.cls_token_segment_id] + unpadded_segment_ids,
            cls_offset=1,
        )


def create_generic_data_row_from_tokens_and_segments(
        guid: str,
        unpadded_tokens: List[str],
        unpadded_segment_ids: List[int],
        label_id: int,
        tokenizer,
        feat_spec: FeaturizationSpec,
        data_row_class):
    input_set = create_input_set_from_tokens_and_segments(
        unpadded_tokens=unpadded_tokens,
        unpadded_segment_ids=unpadded_segment_ids,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )
    return data_row_class(
        guid=guid,
        input_ids=np.array(input_set.input_ids),
        input_mask=np.array(input_set.input_mask),
        segment_ids=np.array(input_set.segment_ids),
        label_id=label_id,
        tokens=unpadded_tokens,
    )


def create_input_set_from_tokens_and_segments(unpadded_tokens: List[str],
                                              unpadded_segment_ids: List[int],
                                              tokenizer,
                                              feat_spec: FeaturizationSpec):
    assert len(unpadded_tokens) == len(unpadded_segment_ids)
    input_ids = tokenizer.convert_tokens_to_ids(unpadded_tokens)
    input_mask = [1] * len(input_ids)
    input_set = pad_features_with_feat_spec(
        input_ids=input_ids,
        input_mask=input_mask,
        unpadded_segment_ids=unpadded_segment_ids,
        feat_spec=feat_spec,
    )
    return input_set


def pad_features_with_feat_spec(input_ids: List[int],
                                input_mask: List[int],
                                unpadded_segment_ids: List[int],
                                feat_spec: FeaturizationSpec):
    return InputSet(
        input_ids=pad_single_with_feat_spec(
            ls=input_ids, feat_spec=feat_spec, pad_idx=feat_spec.pad_token_id,
        ),
        input_mask=pad_single_with_feat_spec(
            ls=input_mask, feat_spec=feat_spec, pad_idx=feat_spec.pad_token_mask_id,
        ),
        segment_ids=pad_single_with_feat_spec(
            ls=unpadded_segment_ids, feat_spec=feat_spec, pad_idx=feat_spec.pad_token_segment_id,
        ),
    )


def pad_single_with_feat_spec(ls: List[int],
                              feat_spec: FeaturizationSpec,
                              pad_idx: int,
                              check=True):
    return pad_to_max_seq_length(
        ls=ls,
        max_seq_length=feat_spec.max_seq_length,
        pad_idx=pad_idx,
        pad_right=not feat_spec.pad_on_left,
        check=check,
    )


def read_tsv(input_file: str, quotechar=None):
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def read_json_lines(input_file: str):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines
