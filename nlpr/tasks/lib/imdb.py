import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from nlpr.tasks.lib.templates.shared import (
    read_json_lines, Task, single_sentence_featurize, TaskTypes,
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
            input_text=tokenizer.tokenize(self.input_text),
            label_id=IMDBTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_text: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.input_text,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class IMDBTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = ["neg", "pos"]
    LABEL_BIMAP = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        raise NotImplementedError()

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            examples.append(Example(
                guid="%s-%s" % (set_type, i),
                input_text=line["input_text"],
                label=line["label"],
            ))
        return examples
