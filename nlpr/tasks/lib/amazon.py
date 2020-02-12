import pandas as pd

import torch
from dataclasses import dataclass
from typing import List

from nlpr.tasks.lib.templates.shared import (
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
            label_id=AmazonPolarityTask.LABEL_BIMAP.a[self.label],
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


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class AmazonPolarityTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = [1, 2]
    LABEL_BIMAP = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self.read_examples(self.train_path, set_type="train")

    def get_val_examples(self):
        return self.read_examples(self.val_path, set_type="val")

    def get_test_examples(self):
        raise NotImplementedError()

    @classmethod
    def read_examples(cls, path, set_type):
        df = pd.read_csv(
            path,
            header=None,
            names=["label", "title", "text"],
        )
        examples = [
            Example(
                guid=f"{set_type}-{i}",
                input_text=row["text"],
                label=row["label"],
            )
            for i, row in df.iterrows()
        ]
        return examples
