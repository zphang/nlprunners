import bs4

import torch
from dataclasses import dataclass
from typing import List

from .shared import (
    read_json_lines, Task, create_generic_data_row_from_tokens_and_segments, add_cls_token,
    TaskTypes,
)
from ..core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from ..utils import truncate_sequences


@dataclass
class Example(BaseExample):
    guid: str
    paragraph: str
    question: str
    answer: str
    label: str
    question_id: int

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            paragraph=tokenizer.tokenize(self.paragraph),
            question=tokenizer.tokenize(self.question),
            answer=tokenizer.tokenize(self.answer),
            label_id=MultiRCTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    paragraph: List
    question: List
    answer: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        paragraph = truncate_sequences(
            tokens_ls=[self.paragraph],
            max_length=feat_spec.max_seq_length - 4 - len(self.question) - len(self.answer),
        )[0]
        unpadded_inputs = add_cls_token(
            unpadded_tokens=(
                paragraph
                + self.question + [tokenizer.sep_token]
                + self.answer + [tokenizer.sep_token],
            ),
            unpadded_segment_ids=(
                [0] * len(paragraph)
                + [1] * len(self.question) + [1]
                + [1] * len(self.answer) + [1]
            ),
            tokenizer=tokenizer,
            feat_spec=feat_spec
        )

        return create_generic_data_row_from_tokens_and_segments(
            guid=self.guid,
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
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
    label_ids: torch.Tensor
    tokens: list

    @classmethod
    def from_data_rows(cls, data_row_ls):
        return Batch(
            input_ids=torch.tensor([f.input_ids for f in data_row_ls], dtype=torch.long),
            input_mask=torch.tensor([f.input_mask for f in data_row_ls], dtype=torch.long),
            segment_ids=torch.tensor([f.segment_ids for f in data_row_ls], dtype=torch.long),
            label_ids=torch.tensor([f.label_id for f in data_row_ls], dtype=torch.long),
            tokens=[f.tokens for f in data_row_ls],
        )


class MultiRCTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.UNDEFINED
    LABELS = [False, True]
    LABEL_BIMAP = labels_to_bimap(LABELS)

    def __init__(self, name, data_dir, filter_sentences=True):
        super().__init__(name=name, data_dir=data_dir)
        self.name = name
        self.data_dir = data_dir
        self.filter_sentences = filter_sentences

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    def _create_examples(self, lines, set_type):
        examples = []
        question_id = 0
        for line in lines:
            soup = bs4.BeautifulSoup(line["paragraph"]["text"], features="lxml")
            sentence_ls = []
            for i, elem in enumerate(soup.html.body.contents):
                if isinstance(elem, bs4.element.NavigableString):
                    sentence_ls.append(str(elem).strip())

            for question_dict in line["paragraph"]["questions"]:
                question = question_dict["question"]
                if self.filter_sentences:
                    paragraph = " ".join(
                        sentence
                        for i, sentence in enumerate(sentence_ls, start=1)
                        if i in question_dict["sentences_used"]
                    )
                else:
                    paragraph = " ".join(sentence_ls)
                for answer_dict in question_dict["answers"]:
                    answer = answer_dict["text"]
                    examples.append(Example(
                        guid="%s-%s" % (set_type, line["idx"]),
                        paragraph=paragraph,
                        question=question,
                        answer=answer,
                        label=answer_dict["isAnswer"] if set_type != "test" else self.LABELS[-1],
                        question_id=question_id,
                    ))
                question_id += 1
        return examples
