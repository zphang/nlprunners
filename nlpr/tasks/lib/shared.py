import csv
import json
from typing import List

from dataclasses import dataclass

from ..utils import truncate_sequences, pad_to_max_seq_length


class Task:

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


def single_sentence_featurize(guid, input_tokens, label_id,
                              tokenizer, feat_spec, data_row_class):
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


def double_sentence_featurize(guid, input_tokens_a, input_tokens_b, label_id,
                              tokenizer, feat_spec, data_row_class):
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


def construct_single_input_tokens_and_segment_ids(input_tokens, tokenizer, feat_spec):
    input_tokens, = truncate_sequences(
        tokens_ls=[input_tokens],
        max_length=feat_spec.max_seq_length - 3,
    )

    return add_cls_token(
        unpadded_tokens=input_tokens + [tokenizer.sep_token],
        unpadded_segment_ids=[0] + [0] * (len(input_tokens)),
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )


def construct_double_input_tokens_and_segment_ids(input_tokens_a, input_tokens_b,
                                                  tokenizer, feat_spec):
    input_tokens_a, input_tokens_b = truncate_sequences(
        tokens_ls=[input_tokens_a, input_tokens_b],
        max_length=feat_spec.max_seq_length - 3,
    )
    unpadded_tokens = (
        input_tokens_a + [tokenizer.sep_token]
        + input_tokens_b + [tokenizer.sep_token]
    )
    unpadded_segment_ids = (
        [0] * len(input_tokens_a) + [0]
        + [1] * len(input_tokens_b) + [1]
    )
    return add_cls_token(
        unpadded_tokens=unpadded_tokens,
        unpadded_segment_ids=unpadded_segment_ids,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )


def add_cls_token(unpadded_tokens, unpadded_segment_ids, tokenizer, feat_spec):
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
            cls_offset=0,
        )


def create_generic_data_row_from_tokens_and_segments(
        guid, unpadded_tokens, unpadded_segment_ids, label_id,
        tokenizer, feat_spec, data_row_class):
    input_set = create_input_set_from_tokens_and_segments(
        unpadded_tokens=unpadded_tokens,
        unpadded_segment_ids=unpadded_segment_ids,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )
    return data_row_class(
        guid=guid,
        input_ids=input_set.input_ids,
        input_mask=input_set.input_mask,
        segment_ids=input_set.segment_ids,
        label_id=label_id,
        tokens=unpadded_tokens,
    )


def create_input_set_from_tokens_and_segments(unpadded_tokens, unpadded_segment_ids,
                                              tokenizer, feat_spec):
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


def pad_features_with_feat_spec(input_ids, input_mask, unpadded_segment_ids, feat_spec):
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


def pad_single_with_feat_spec(ls, feat_spec, pad_idx, check=True):
    return pad_to_max_seq_length(
        ls=ls,
        max_seq_length=feat_spec.max_seq_length,
        pad_idx=pad_idx,
        pad_right=not feat_spec.pad_on_left,
        check=check,
    )


def read_tsv(input_file, quotechar=None):
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def read_json_lines(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines
