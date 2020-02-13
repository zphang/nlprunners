import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from nlpr.tasks.lib.templates.shared import (
    read_json_lines, Task, create_input_set_from_tokens_and_segments, add_cls_token, TaskTypes,
)
from ..core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from ..utils import truncate_sequences


@dataclass
class Example(BaseExample):
    guid: str
    question: str
    choice_list: List[str]
    label: int

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            question=tokenizer.tokenize(self.question),
            choice_list=[
                tokenizer.tokenize(choice)
                for choice in self.choice_list
            ],
            label_id=CommonsenseQATask.CHOICE_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    question: List
    choice_list: List[List]
    label_id: int

    def featurize(self, tokenizer, feat_spec):

        if feat_spec.sep_token_extra:
            maybe_extra_sep = [tokenizer.sep_token]
            maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
            special_tokens_count = 4  # CLS, SEP-SEP, SEP
        else:
            maybe_extra_sep = []
            maybe_extra_sep_segment_id = []
            special_tokens_count = 3  # CLS, SEP, SEP

        input_set_ls = []
        unpadded_inputs_ls = []
        for choice in self.choice_list:
            question, choice = truncate_sequences(
                tokens_ls=[self.question, choice],
                max_length=feat_spec.max_seq_length - special_tokens_count,
            )
            unpadded_inputs = add_cls_token(
                unpadded_tokens=(
                    # question
                    question + [tokenizer.sep_token] + maybe_extra_sep
                    # choice
                    + choice + [tokenizer.sep_token]
                ),
                unpadded_segment_ids=(
                    # premise
                    [feat_spec.sequence_a_segment_id] * (len(question) + 1)
                    + maybe_extra_sep_segment_id
                    # choice + sep
                    + [feat_spec.sequence_b_segment_id] * (len(choice) + 1)
                ),
                tokenizer=tokenizer,
                feat_spec=feat_spec,
            )
            input_set = create_input_set_from_tokens_and_segments(
                unpadded_tokens=unpadded_inputs.unpadded_tokens,
                unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
                tokenizer=tokenizer,
                feat_spec=feat_spec
            )
            input_set_ls.append(input_set)
            unpadded_inputs_ls.append(unpadded_inputs)

        return DataRow(
            guid=self.guid,
            input_ids=np.stack([input_set.input_ids for input_set in input_set_ls]),
            input_mask=np.stack([input_set.input_mask for input_set in input_set_ls]),
            segment_ids=np.stack([input_set.segment_ids for input_set in input_set_ls]),
            label_id=self.label_id,
            tokens_list=[unpadded_inputs.unpadded_tokens for unpadded_inputs in unpadded_inputs_ls],
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray  # Multiple
    input_mask: np.ndarray  # Multiple
    segment_ids: np.ndarray  # Multiple
    label_id: int
    tokens_list: List[List]  # Multiple


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens_list: List


class CommonsenseQATask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.MULTIPLE_CHOICE
    NUM_CHOICES = 5
    LABELS = [0, 1]
    LABEL_BIMAP = labels_to_bimap(LABELS)

    CHOICE_KEYS = ["A", "B", "C", "D", "E"]
    CHOICE_BIMAP = labels_to_bimap(CHOICE_KEYS)

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            choice_dict = {
                elem["label"]: elem["text"]
                for elem in line["question"]["choices"]
            }
            examples.append(Example(
                guid="%s-%s" % (set_type, i),
                question=line["question"]["stem"],
                choice_list=[choice_dict[key] for key in cls.CHOICE_KEYS],
                label=line["answerKey"],
            ))
        return examples
