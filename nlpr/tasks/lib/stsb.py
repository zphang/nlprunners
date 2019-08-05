import torch
from dataclasses import dataclass
from typing import List

from .shared import (
    read_json_lines, Task, construct_double_input_tokens_and_segment_ids,
    create_input_set_from_tokens_and_segments, TaskTypes,
)
from ..core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin


@dataclass
class Example(BaseExample):
    guid: str
    text_a: str
    text_b: str
    label: float

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text_a=tokenizer.tokenize(self.text_a),
            text_b=tokenizer.tokenize(self.text_b),
            label=self.label,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text_a: List
    text_b: List
    label: float

    def featurize(self, tokenizer, feat_spec):
        unpadded_inputs = construct_double_input_tokens_and_segment_ids(
            input_tokens_a=self.text_a,
            input_tokens_b=self.text_b,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        return DataRow(
            guid=self.guid,
            input_ids=input_set.input_ids,
            input_mask=input_set.input_mask,
            segment_ids=input_set.segment_ids,
            label=self.label,
            tokens=unpadded_inputs.unpadded_tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: list
    input_mask: list
    segment_ids: list
    label: float
    tokens: list

    def get_tokens(self):
        return [self.tokens]


@dataclass
class Batch(BatchMixin):
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    segment_ids: torch.Tensor
    label: torch.Tensor
    tokens: list

    @classmethod
    def from_data_rows(cls, data_row_ls):
        return Batch(
            input_ids=torch.tensor([f.input_ids for f in data_row_ls], dtype=torch.long),
            input_mask=torch.tensor([f.input_mask for f in data_row_ls], dtype=torch.long),
            segment_ids=torch.tensor([f.segment_ids for f in data_row_ls], dtype=torch.long),
            label=torch.tensor([f.label_id for f in data_row_ls], dtype=torch.float),
            tokens=[f.tokens for f in data_row_ls],
        )


class StsbTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.REGRESSION

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            examples.append(Example(
                guid="%s-%s" % (set_type, i),
                text_a=line["text_a"],
                text_b=line["text_b"],
                label=float(line["label"]) if set_type != "test" else 0,
            ))
        return examples
