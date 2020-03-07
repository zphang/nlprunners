import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from nlpr.tasks.lib.templates.shared import (
    Task, TaskTypes,
    construct_single_input_tokens_and_segment_ids,
    create_input_set_from_tokens_and_segments,
)
from ..core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin


@dataclass
class Example(BaseExample):
    guid: str
    text: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            input_tokens=tokenizer.tokenize(self.text),
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_tokens: List

    def featurize(self, tokenizer, feat_spec):
        unpadded_inputs = construct_single_input_tokens_and_segment_ids(
            input_tokens=self.input_tokens,
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
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            tokens=unpadded_inputs.unpadded_tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    tokens: list


class MLMTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.MASKED_LANGUAGE_MODELING

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train", return_generator=True)

    def get_val_examples(self):
        return self._create_examples(path=self.val_path, set_type="val", return_generator=True)

    def get_test_examples(self):
        return self._create_examples(path=self.test_path, set_type="test", return_generator=True)

    @classmethod
    def _get_examples_generator(cls, path, set_type):
        for (i, line) in enumerate(path):
            yield Example(
                guid="%s-%s" % (set_type, i),
                text=line.strip(),
            )

    @classmethod
    def _create_examples(cls, path, set_type, return_generator):
        generator = cls._get_examples_generator(path=path, set_type=set_type)
        if return_generator:
            return generator
        else:
            return list(generator)
