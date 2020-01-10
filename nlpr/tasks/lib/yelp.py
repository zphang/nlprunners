import pandas as pd

import torch
from dataclasses import dataclass
from typing import List

import pyutils.io as io

from .shared import (
    Task, single_sentence_featurize, TaskTypes,
)
from ..core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap


@dataclass
class Example(BaseExample):
    guid: str
    input_text: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            input_tokens=tokenizer.tokenize(self.input_text),
            label_id=YelpPolarityTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_tokens: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.input_tokens,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: list
    input_mask: list
    segment_ids: list
    label_id: int
    tokens: list

    def get_tokens(self):
        return [self.tokens]


@dataclass
class Batch(BatchMixin):
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    segment_ids: torch.Tensor
    label_id: torch.Tensor
    tokens: list

    @classmethod
    def from_data_rows(cls, data_row_ls):
        return Batch(
            input_ids=torch.tensor([f.input_ids for f in data_row_ls], dtype=torch.long),
            input_mask=torch.tensor([f.input_mask for f in data_row_ls], dtype=torch.long),
            segment_ids=torch.tensor([f.segment_ids for f in data_row_ls], dtype=torch.long),
            label_id=torch.tensor([f.label_id for f in data_row_ls], dtype=torch.long),
            tokens=[f.tokens for f in data_row_ls],
        )


class YelpPolarityTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = [1, 2, 3, 4, 5]
    LABEL_BIMAP = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self.read_examples(self.train_path, set_type="train")

    def get_val_examples(self):
        return self.read_examples(self.val_path, set_type="val")

    def get_test_examples(self):
        raise NotImplementedError()

    @classmethod
    def read_examples(cls, path, set_type):
        if path.endswith(".csv"):
            df = pd.read_csv(
                path,
                header=None,
                names=["label", "text"],
            )
            iterator = df.iterrows()
        elif path.endswith(".jsonl"):
            iterator = enumerate(io.read_jsonl(path))
        else:
            raise RuntimeError(path)
        examples = [
            Example(
                guid=f"{set_type}-{i}",
                input_text=row["text"],
                label=row["label"],
            )
            for i, row in iterator
        ]
        return examples
