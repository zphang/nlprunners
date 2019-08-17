import torch
from dataclasses import dataclass
from typing import List

from .shared import (
    read_json_lines, Task, create_input_set_from_tokens_and_segments, add_cls_token, TaskTypes,
)
from ..core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from ..utils import truncate_sequences, convert_word_idx_for_bert_tokens


@dataclass
class Example(BaseExample):
    guid: str
    sent1: str
    sent2: str
    word: str
    sent1_idx: int
    sent2_idx: int
    label: str

    def tokenize(self, tokenizer):
        sent1_tokens = tokenizer.tokenize(self.sent1)
        sent2_tokens = tokenizer.tokenize(self.sent2)
        sent1_span = convert_word_idx_for_bert_tokens(
            text=self.sent1,
            bert_tokens=sent1_tokens,
            word_idx_ls=[self.sent1_idx],
            check=False,
        )[0]
        sent2_span = convert_word_idx_for_bert_tokens(
            text=self.sent2,
            bert_tokens=sent2_tokens,
            word_idx_ls=[self.sent2_idx],
            check=False,
        )[0]
        return TokenizedExample(
            guid=self.guid,
            sent1_tokens=tokenizer.tokenize(self.sent1),
            sent2_tokens=tokenizer.tokenize(self.sent2),
            word=tokenizer.tokenize(self.word),  # might be more than one token
            sent1_span=sent1_span,
            sent2_span=sent2_span,
            label_id=WiCTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    sent1_tokens: List
    sent2_tokens: List
    word: List
    sent1_span: List
    sent2_span: List
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

        sent1_tokens, sent2_tokens = truncate_sequences(
            tokens_ls=[self.sent1_tokens, self.sent2_tokens],
            max_length=feat_spec.max_seq_length - len(self.word) - special_tokens_count,
        )

        unpadded_tokens = (
            self.word + [tokenizer.sep_token] + maybe_extra_sep
            + sent1_tokens + [tokenizer.sep_token] + maybe_extra_sep
            + sent2_tokens + [tokenizer.sep_token]
        )
        # Don't have a choice here -- just leave words as part of sent1
        unpadded_segment_ids = (
                [feat_spec.sequence_a_segment_id] * (len(self.word) + 1)
                + maybe_extra_sep_segment_id
                + [feat_spec.sequence_a_segment_id] * (len(sent1_tokens) + 2)
                + maybe_extra_sep_segment_id
                + [feat_spec.sequence_b_segment_id] * (len(sent2_tokens) + 1)
        )

        unpadded_inputs = add_cls_token(
            unpadded_tokens=unpadded_tokens,
            unpadded_segment_ids=unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        end_span_offset = -1  # Inclusive span
        word_sep_offset = 1
        sent1_sep_offset = 1

        sent1_span = [
            self.sent1_span[0] + unpadded_inputs.cls_offset + word_sep_offset
            + len(self.word),
            self.sent1_span[1] + unpadded_inputs.cls_offset + word_sep_offset
            + len(self.word) + end_span_offset,
        ]
        sent2_span = [
            self.sent2_span[0] + unpadded_inputs.cls_offset + word_sep_offset + sent1_sep_offset
            + len(self.word) + len(sent1_tokens),
            self.sent2_span[1] + unpadded_inputs.cls_offset + word_sep_offset + sent1_sep_offset
            + len(self.word) + len(sent1_tokens) + end_span_offset,
        ]

        return DataRow(
            guid=self.guid,
            input_ids=input_set.input_ids,
            input_mask=input_set.input_mask,
            segment_ids=input_set.segment_ids,
            sent1_span=sent1_span,
            sent2_span=sent2_span,
            label_id=self.label_id,
            tokens=unpadded_inputs.unpadded_tokens,
            word=self.word,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: List
    input_mask: List
    segment_ids: List
    sent1_span: List
    sent2_span: List
    label_id: int
    tokens: List
    word: List

    def get_tokens(self):
        return [self.tokens]


@dataclass
class Batch(BatchMixin):
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    segment_ids: torch.Tensor
    sent1_span: torch.Tensor
    sent2_span: torch.Tensor
    label_ids: torch.Tensor
    tokens: List
    word: List

    @classmethod
    def from_data_rows(cls, data_row_ls):
        return Batch(
            input_ids=torch.tensor([f.input_ids for f in data_row_ls], dtype=torch.long),
            input_mask=torch.tensor([f.input_mask for f in data_row_ls], dtype=torch.long),
            segment_ids=torch.tensor([f.segment_ids for f in data_row_ls], dtype=torch.long),
            sent1_span=torch.tensor([f.sent1_span for f in data_row_ls], dtype=torch.long),
            sent2_span=torch.tensor([f.sent2_span for f in data_row_ls], dtype=torch.long),
            label_ids=torch.tensor([f.label_id for f in data_row_ls], dtype=torch.long),
            tokens=[f.tokens for f in data_row_ls],
            word=[f.word for f in data_row_ls],
        )


class WiCTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.UNDEFINED
    LABELS = [False, True]
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
                sent1=line["sentence1"],
                sent2=line["sentence2"],
                word=line["word"],
                sent1_idx=int(line["sentence1_idx"]),
                sent2_idx=int(line["sentence2_idx"]),
                label=line["label"] if set_type != "test" else cls.LABELS[-1],
            ))
        return examples
