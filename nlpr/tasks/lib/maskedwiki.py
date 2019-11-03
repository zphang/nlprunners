import torch
from dataclasses import dataclass
from typing import List
import pyutils.io as io

from .shared import (
    read_json_lines, Task,
    construct_single_input_tokens_and_segment_ids,
    create_input_set_from_tokens_and_segments,
    TaskTypes, Span
)
from ..core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap


@dataclass
class Example(BaseExample):
    guid: str
    text_a: str
    text_b: str
    option_0: str
    option_1: str
    label: str

    def tokenize(self, tokenizer):
        t1 = tokenizer.tokenize(" ".join([self.text_a, self.option_0, self.text_b]))
        t1a = tokenizer.tokenize(self.text_a)
        t1b = tokenizer.tokenize(" ".join([self.text_a, self.option_0]))

        t2 = tokenizer.tokenize(" ".join([self.text_a, self.option_1, self.text_b]))
        t2a = tokenizer.tokenize(self.text_a)
        t2b = tokenizer.tokenize(" ".join([self.text_a, self.option_1]))

        return TokenizedExample(
            guid=self.guid,
            option_1_tokens=t1,
            option_2_tokens=t2,
            option_1_span=Span(len(t1a), len(t1b)),
            option_2_span=Span(len(t2a), len(t2b)),
            label_id=MaskedWikiTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    option_1_tokens: List
    option_2_tokens: List
    option_1_span: Span
    option_2_span: Span
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        unpadded_inputs_1 = construct_single_input_tokens_and_segment_ids(
            input_tokens=self.option_1_tokens,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        unpadded_inputs_2 = construct_single_input_tokens_and_segment_ids(
            input_tokens=self.option_2_tokens,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        input_set_1 = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs_1.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs_1.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        input_set_2 = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs_2.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs_2.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        option_1_span = self.option_1_span
        option_2_span = self.option_2_span
        if not feat_spec.cls_token_at_end:
            option_1_span = option_1_span.add(1)
            option_2_span = option_2_span.add(1)

        return DataRow(
            guid=self.guid,
            input_ids_1=input_set_1.input_ids,
            input_mask_1=input_set_1.input_mask,
            segment_ids_1=input_set_1.segment_ids,
            input_ids_2=input_set_2.input_ids,
            input_mask_2=input_set_2.input_mask,
            segment_ids_2=input_set_2.segment_ids,
            option_1_span=option_1_span,
            option_2_span=option_2_span,
            label_id=self.label_id,
            tokens_1=unpadded_inputs_1.unpadded_tokens,
            tokens_2=unpadded_inputs_2.unpadded_tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids_1: list
    input_mask_1: list
    segment_ids_1: list
    input_ids_2: list
    input_mask_2: list
    segment_ids_2: list
    option_1_span: Span
    option_2_span: Span
    label_id: int
    tokens_1: list
    tokens_2: list

    def get_tokens(self):
        return [self.tokens_1, self.tokens_2]


@dataclass
class Batch(BatchMixin):
    input_ids_1: torch.Tensor
    input_mask_1: torch.Tensor
    segment_ids_1: torch.Tensor
    input_ids_2: torch.Tensor
    input_mask_2: torch.Tensor
    segment_ids_2: torch.Tensor
    label_ids: torch.Tensor
    option_1_span: torch.Tensor
    option_2_span: torch.Tensor
    tokens_1: list
    tokens_2: list

    @classmethod
    def from_data_rows(cls, data_row_ls: List[DataRow]):
        return Batch(
            input_ids_1=torch.tensor([f.input_ids_1 for f in data_row_ls], dtype=torch.long),
            input_mask_1=torch.tensor([f.input_mask_1 for f in data_row_ls], dtype=torch.long),
            segment_ids_1=torch.tensor([f.segment_ids_1 for f in data_row_ls], dtype=torch.long),
            input_ids_2=torch.tensor([f.input_ids_2 for f in data_row_ls], dtype=torch.long),
            input_mask_2=torch.tensor([f.input_mask_2 for f in data_row_ls], dtype=torch.long),
            segment_ids_2=torch.tensor([f.segment_ids_2 for f in data_row_ls], dtype=torch.long),
            option_1_span=torch.tensor([f.option_1_span for f in data_row_ls], dtype=torch.long),
            option_2_span=torch.tensor([f.option_2_span for f in data_row_ls], dtype=torch.long),
            label_ids=torch.tensor([f.label_id for f in data_row_ls], dtype=torch.long),
            tokens_1=[f.tokens_1 for f in data_row_ls],
            tokens_2=[f.tokens_2 for f in data_row_ls],
        )


class MaskedWikiTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.SPAN_CHOICE_PROB_TASK
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
        for (i, line) in enumerate(lines):
            examples.append(Example(
                guid="%s-%s" % (set_type, i),
                text_a=line["text_a"],
                text_b=line["text_b"],
                option_0=line["option_0"],
                option_1=line["option_1"],
                label=line["label"] if set_type != "test" else cls.LABELS[-1],
            ))
        return examples


# ==== Utilities ==== #


def grouper(n, iterable):
    args = [iter(iterable)] * n
    return zip(*args)


def map_strip(ls):
    return [x.strip() for x in ls]


def process_raw(input_path, output_path):
    lines = io.read_file_lines(input_path)
    examples = []
    for text, mask, options, correct, blank in grouper(5, lines):
        text, mask, options, correct = map_strip([text, mask, options, correct])
        assert mask == "[MASK]"
        assert options.count(",") == 1
        assert blank == "\n"
        text_split = map_strip(text.split(mask))
        options = options.split(",")
        assert len(text_split) == 2
        if correct == options[0]:
            label = 0
        elif correct == options[1]:
            label = 1
        else:
            raise RuntimeError()
        examples.append({
            "label": label,
            "option_0": options[0],
            "option_1": options[1],
            "text_a": text_split[0],
            "text_b": text_split[1],
        })
    io.write_jsonl(examples, output_path)
