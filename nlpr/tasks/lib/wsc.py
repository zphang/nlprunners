import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from .shared import (
    read_json_lines, Task, create_input_set_from_tokens_and_segments, add_cls_token, TaskTypes,
)
from ..core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from ..utils import truncate_sequences, convert_char_span_for_bert_tokens


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    span1_idx: int
    span2_idx: int
    span1_text: str
    span2_text: str
    label: str

    def tokenize(self, tokenizer):
        text_tokens = self.text.split()
        bert_tokens = tokenizer.tokenize(self.text)
        if self.span1_idx == 0:
            span1_start_idx = 0
        else:
            span1_start_idx = len(" ".join(text_tokens[:self.span1_idx]) + " ")
        if self.span2_idx == 0:
            span2_start_idx = 0
        else:
            span2_start_idx = len(" ".join(text_tokens[:self.span2_idx]) + " ")
        span1_span = convert_char_span_for_bert_tokens(
            text=self.text,
            bert_tokens=bert_tokens,
            span_ls=[[span1_start_idx, self.span1_text]],
            check=False,
        )[0]
        span2_span = convert_char_span_for_bert_tokens(
            text=self.text,
            bert_tokens=bert_tokens,
            span_ls=[[span2_start_idx, self.span2_text]],
            check=False,
        )[0]
        return TokenizedExample(
            guid=self.guid,
            tokens=bert_tokens,
            span1_span=span1_span,
            span2_span=span2_span,
            span1_text=self.span1_text,
            span2_text=self.span2_text,
            label_id=WSCTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    tokens: List
    span1_span: List
    span2_span: List
    span1_text: str
    span2_text: str
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        if feat_spec.sep_token_extra:
            maybe_extra_sep = [tokenizer.sep_token]
            maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
            special_tokens_count = 3  # CLS, SEP-SEP  (ok this is a little weird, let's leave it for now)
        else:
            maybe_extra_sep = []
            maybe_extra_sep_segment_id = []
            special_tokens_count = 2  # CLS, SEP

        tokens = truncate_sequences(
            tokens_ls=[self.tokens],
            max_length=feat_spec.max_seq_length - special_tokens_count,
        )[0]

        unpadded_tokens = tokens + [tokenizer.sep_token] + maybe_extra_sep
        unpadded_segment_ids = (
            [feat_spec.sequence_a_segment_id] * (len(self.tokens) + 1)
            + maybe_extra_sep_segment_id
        )

        unpadded_inputs = add_cls_token(
            unpadded_tokens=unpadded_tokens,
            unpadded_segment_ids=unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        end_span_offset = -1  # Inclusive span

        span1_span = [
            self.span1_span[0] + unpadded_inputs.cls_offset,
            self.span1_span[1] + unpadded_inputs.cls_offset + end_span_offset,
        ]
        span2_span = [
            self.span2_span[0] + unpadded_inputs.cls_offset,
            self.span2_span[1] + unpadded_inputs.cls_offset + end_span_offset,
        ]

        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            span1_span=np.array(span1_span),
            span2_span=np.array(span2_span),
            label_id=self.label_id,
            tokens=unpadded_inputs.unpadded_tokens,
            span1_text=self.span1_text,
            span2_text=self.span2_text,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    span1_span: np.ndarray
    span2_span: np.ndarray
    label_id: int
    tokens: List
    span1_text: str
    span2_text: str

    def get_tokens(self):
        return [self.tokens]


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    span1_span: torch.LongTensor
    span2_span: torch.LongTensor
    label_id: torch.LongTensor
    tokens: List
    span1_text: List
    span2_text: List


class WSCTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.SPAN_COMPARISON_CLASSIFICATION
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
                text=line["text"],
                span1_idx=line["target"]["span1_index"],
                span2_idx=line["target"]["span2_index"],
                span1_text=line["target"]["span1_text"],
                span2_text=line["target"]["span2_text"],
                label=line["label"] if set_type != "test" else cls.LABELS[-1],
            ))
        return examples
