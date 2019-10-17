import torch
from dataclasses import dataclass
from typing import List

from .shared import (
    read_json_lines, Task, add_cls_token, TaskTypes, create_input_set_from_tokens_and_segments,
)
from ..core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from ..utils import truncate_sequences


@dataclass
class Example(BaseExample):
    guid: str
    input_obs1: str
    input_hyp1: str
    input_hyp2: str
    input_obs2: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            input_obs1=tokenizer.tokenize(self.input_obs1),
            input_hyp1=tokenizer.tokenize(self.input_hyp1),
            input_hyp2=tokenizer.tokenize(self.input_hyp2),
            input_obs2=tokenizer.tokenize(self.input_obs2),
            label_id=AnliTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_obs1: List
    input_hyp1: List
    input_hyp2: List
    input_obs2: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        if feat_spec.sep_token_extra:
            maybe_extra_sep = [tokenizer.sep_token]
            maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
            special_tokens_count = 6  # CLS, SEP-SEP, SEP-SEP, SEP
        else:
            maybe_extra_sep = []
            maybe_extra_sep_segment_id = []
            special_tokens_count = 4  # CLS, SEP, SEP, SEP

        input_obs1_a, input_hyp1_a, input_obs2_a = truncate_sequences(
            tokens_ls=[self.input_obs1, self.input_hyp1, self.input_obs2],
            max_length=feat_spec.max_seq_length - special_tokens_count - 1,
            # -1 for self.question
        )
        input_obs1_b, input_hyp2_b, input_obs2_b = truncate_sequences(
            tokens_ls=[self.input_obs1, self.input_hyp2, self.input_obs2],
            max_length=feat_spec.max_seq_length - special_tokens_count - 1,
            # -1 for self.question
        )

        unpadded_inputs_1 = add_cls_token(
            unpadded_tokens=(
                input_obs1_a + [tokenizer.sep_token] + maybe_extra_sep
                + input_hyp1_a + [tokenizer.sep_token] + maybe_extra_sep
                + input_obs2_a + [tokenizer.sep_token]
            ),
            unpadded_segment_ids=(
                # question + sep(s)
                [feat_spec.sequence_a_segment_id] * (len(input_obs1_a) + 1)
                + maybe_extra_sep_segment_id
                # premise + sep(s)
                + [feat_spec.sequence_a_segment_id] * (len(input_hyp1_a) + 1)
                + maybe_extra_sep_segment_id
                # choice + sep
                + [feat_spec.sequence_b_segment_id] * (len(input_obs2_a) + 1)
            ),
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        unpadded_inputs_2 = add_cls_token(
            unpadded_tokens=(
                input_obs1_b + [tokenizer.sep_token] + maybe_extra_sep
                + input_hyp2_b + [tokenizer.sep_token] + maybe_extra_sep
                + input_obs2_b + [tokenizer.sep_token]
            ),
            unpadded_segment_ids=(
                # question + sep(s)
                [feat_spec.sequence_a_segment_id] * (len(input_obs1_b) + 1)
                + maybe_extra_sep_segment_id
                # premise + sep(s)
                + [feat_spec.sequence_a_segment_id] * (len(input_hyp2_b) + 1)
                + maybe_extra_sep_segment_id
                # choice + sep
                + [feat_spec.sequence_b_segment_id] * (len(input_obs2_b) + 1)
            ),
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        input_set1 = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs_1.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs_1.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec
        )
        input_set2 = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs_2.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs_2.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec
        )
        return DataRow(
            guid=self.guid,
            input_ids1=input_set1.input_ids,
            input_mask1=input_set1.input_mask,
            segment_ids1=input_set1.segment_ids,
            input_ids2=input_set2.input_ids,
            input_mask2=input_set2.input_mask,
            segment_ids2=input_set2.segment_ids,
            label_id=self.label_id,
            tokens1=unpadded_inputs_1.unpadded_tokens,
            tokens2=unpadded_inputs_2.unpadded_tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids1: list
    input_mask1: list
    segment_ids1: list
    input_ids2: list
    input_mask2: list
    segment_ids2: list
    label_id: int
    tokens1: list
    tokens2: list

    def get_tokens(self):
        return [self.tokens1, self.tokens2]


@dataclass
class Batch(BatchMixin):
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    segment_ids: torch.Tensor
    label_ids: torch.Tensor
    tokens1: list
    tokens2: list

    @classmethod
    def from_data_rows(cls, data_row_ls):
        return Batch(
            input_ids=torch.tensor([
                [f.input_ids1, f.input_ids2]
                for f in data_row_ls
            ], dtype=torch.long),
            input_mask=torch.tensor([
                [f.input_mask1, f.input_mask2]
                for f in data_row_ls
            ], dtype=torch.long),
            segment_ids=torch.tensor([
                [f.segment_ids1, f.segment_ids2]
                for f in data_row_ls
            ], dtype=torch.long),
            label_ids=torch.tensor([f.label_id for f in data_row_ls], dtype=torch.long),
            tokens1=[f.tokens1 for f in data_row_ls],
            tokens2=[f.tokens2 for f in data_row_ls],
        )


class AnliTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.MULTIPLE_CHOICE
    NUM_CHOICES = 2
    LABELS = [1, 2]
    LABEL_BIMAP = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(
            lines=read_json_lines(self.path_dict["train_inputs"]),
            labels=self._read_labels(self.path_dict["train_labels"]),
            set_type="train"
        )

    def get_val_examples(self):
        return self._create_examples(
            lines=read_json_lines(self.path_dict["val_inputs"]),
            labels=self._read_labels(self.path_dict["val_labels"]),
            set_type="val",
        )

    def get_test_examples(self):
        raise NotImplementedError()

    @classmethod
    def _create_examples(cls, lines, labels, set_type):
        examples = []
        for (i, (line, label)) in enumerate(zip(lines, labels)):
            examples.append(Example(
                guid="%s-%s" % (set_type, i),
                input_obs1=line["obs1"],
                input_hyp1=line["hyp1"],
                input_hyp2=line["hyp2"],
                input_obs2=line["obs2"],
                label=label,
            ))
        return examples

    @classmethod
    def _read_labels(cls, path):
        with open(path) as f:
            return [int(i) for i in f.read().split()]
