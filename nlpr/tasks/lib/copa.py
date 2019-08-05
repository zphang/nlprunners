import torch
from dataclasses import dataclass
from typing import List

from .shared import read_json_lines, Task, create_input_set_from_tokens_and_segments, add_cls_token
from ..core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from ..utils import truncate_sequences


@dataclass
class Example(BaseExample):
    guid: str
    input_premise: str
    input_choice1: str
    input_choice2: str
    question: str
    label: int

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            input_premise=tokenizer.tokenize(self.input_premise),
            input_choice1=tokenizer.tokenize(self.input_choice1),
            input_choice2=tokenizer.tokenize(self.input_choice2),
            # Safe assumption that question is a single word
            question=tokenizer.tokenize(self.question)[0],
            label_id=CopaTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_premise: List
    input_choice1: List
    input_choice2: List
    question: str  # Safe assumption that question is a single word
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        input_premise, input_choice1 = truncate_sequences(
            tokens_ls=[self.input_premise, self.input_choice1],
            max_length=feat_spec.max_seq_length - 5,
        )
        input_premise, input_choice2 = truncate_sequences(
            tokens_ls=[self.input_premise, self.input_choice2],
            max_length=feat_spec.max_seq_length - 5,
        )

        unpadded_inputs_1 = add_cls_token(
            unpadded_tokens=(
                [self.question] + [tokenizer.sep_token]
                + input_premise + [tokenizer.sep_token]
                + input_choice1 + [tokenizer.sep_token]
            ),
            unpadded_segment_ids=(
                [0] * (len(input_premise) + 3)  # includes question and 2 SEPs
                + [1] * (len(input_choice1) + 1)  # includes SEP
            ),
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        unpadded_inputs_2 = add_cls_token(
            unpadded_tokens=(
                [self.question] + [tokenizer.sep_token]
                + input_premise + [tokenizer.sep_token]
                + input_choice2 + [tokenizer.sep_token]
            ),
            unpadded_segment_ids=(
                [0] * (len(input_premise) + 3)  # includes question and 2 SEPs
                + [1] * (len(input_choice2) + 1)  # includes SEP
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
    input_ids1: torch.Tensor
    input_mask1: torch.Tensor
    segment_ids1: torch.Tensor
    input_ids2: torch.Tensor
    input_mask2: torch.Tensor
    segment_ids2: torch.Tensor
    label_ids: torch.Tensor
    tokens1: list
    tokens2: list

    @classmethod
    def from_data_rows(cls, data_row_ls):
        return Batch(
            input_ids1=torch.tensor([f.input_ids1 for f in data_row_ls], dtype=torch.long),
            input_mask1=torch.tensor([f.input_mask1 for f in data_row_ls], dtype=torch.long),
            segment_ids1=torch.tensor([f.segment_ids1 for f in data_row_ls], dtype=torch.long),
            input_ids2=torch.tensor([f.input_ids2 for f in data_row_ls], dtype=torch.long),
            input_mask2=torch.tensor([f.input_mask2 for f in data_row_ls], dtype=torch.long),
            segment_ids2=torch.tensor([f.segment_ids2 for f in data_row_ls], dtype=torch.long),
            label_ids=torch.tensor([f.label_id for f in data_row_ls], dtype=torch.long),
            tokens1=[f.tokens1 for f in data_row_ls],
            tokens2=[f.tokens2 for f in data_row_ls],
        )


class CopaTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    LABELS = [0, 1]
    LABEL_BIMAP = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for line in lines:
            examples.append(Example(
                guid="%s-%s" % (set_type, line["idx"]),
                input_premise=line["premise"],
                input_choice1=line["choice1"],
                input_choice2=line["choice2"],
                question=line["question"],
                label=line["label"] if set_type != "test" else cls.LABELS[-1],
            ))
        return examples
