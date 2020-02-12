import json
import numpy as np
import tqdm

import torch
from dataclasses import dataclass
from typing import Union

from nlpr.tasks.lib.templates.shared import Task, TaskTypes
from ..core import BaseExample, BaseDataRow, BatchMixin, FeaturizationSpec
from transformers.tokenization_bert import whitespace_tokenize
from nlpr.constants import PHASE

import logging
logger = logging.getLogger(__name__)


@dataclass
class Example(BaseExample):
    qas_id: str
    question_text: str
    context_text: str
    answer_text: str
    start_position_character: int
    title: str
    answers: list
    is_impossible: bool

    # ===
    doc_tokens: Union[list, None] = None
    char_to_word_offset: Union[list, None] = None
    start_position: int = 0
    end_position: int = 0

    def __post_init__(self):
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if self.start_position_character is not None and not self.is_impossible:
            self.start_position = char_to_word_offset[self.start_position_character]
            self.end_position = char_to_word_offset[min(
                self.start_position_character + len(self.answer_text) - 1,
                len(char_to_word_offset) - 1
            )]

    def tokenize(self, tokenizer):
        raise NotImplementedError("SQuaD is weird")

    def to_feature_list(self, tokenizer, feat_spec: FeaturizationSpec,
                        max_seq_length, doc_stride, max_query_length,
                        set_type):
        is_training = set_type == PHASE.TRAIN
        features = []
        if is_training and not self.is_impossible:
            # Get start and end position
            start_position = self.start_position
            end_position = self.end_position

            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(self.doc_tokens[start_position: (end_position + 1)])
            cleaned_answer_text = " ".join(whitespace_tokenize(self.answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                return []

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(self.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if is_training and not self.is_impossible:
            tok_start_position = orig_to_tok_index[self.start_position]
            if self.end_position < len(self.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[self.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, self.answer_text
            )

        spans = []

        truncated_query = tokenizer.encode(self.question_text, add_special_tokens=False, max_length=max_query_length)
        sequence_added_tokens = (
            tokenizer.max_len - tokenizer.max_len_single_sentence + 1
            if "roberta" in str(type(tokenizer))
            else tokenizer.max_len - tokenizer.max_len_single_sentence
        )
        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):

            encoded_dict = tokenizer.encode_plus(
                truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
                span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                pad_to_max_length=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
            )

            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict:
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = (
                    j
                    if tokenizer.padding_side == "left"
                    else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                )
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context

        for span in spans:
            # Identify the position of the CLS token
            cls_index = span["input_ids"].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = np.array(span["token_type_ids"])

            p_mask = np.minimum(p_mask, 1)

            if tokenizer.padding_side == "right":
                # Limit positive values to one
                p_mask = 1 - p_mask

            p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

            # Set the CLS index to '0'
            p_mask[cls_index] = 0

            span_is_impossible = self.is_impossible
            start_position = 0
            end_position = 0
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                out_of_span = False

                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = cls_index
                    end_position = cls_index
                    span_is_impossible = True
                else:
                    if tokenizer.padding_side == "left":
                        doc_offset = 0
                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens

                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            features.append(
                DataRow(
                    unique_id=f"{self.qas_id}-{len(features)}",
                    qas_id=self.qas_id,
                    tokens=span["tokens"],
                    token_to_orig_map=span["token_to_orig_map"],
                    token_is_max_context=span["token_is_max_context"],
                    input_ids=np.array(span["input_ids"]),
                    input_mask=np.array(span["attention_mask"]),
                    segment_ids=np.array(span["token_type_ids"]),
                    cls_index=np.array(cls_index),
                    p_mask=np.array(p_mask.tolist()),
                    paragraph_len=span["paragraph_len"],
                    start_position=start_position,
                    end_position=end_position,
                )
            )
        return features


@dataclass
class DataRow(BaseDataRow):
    unique_id: str
    qas_id: str
    tokens: list
    token_to_orig_map: dict
    token_is_max_context: dict
    input_ids: np.array
    input_mask: np.array
    segment_ids: np.array
    cls_index: np.array
    p_mask: np.array
    paragraph_len: int
    start_position: int
    end_position: int


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    start_position: torch.LongTensor
    end_position: torch.LongTensor
    cls_index: torch.LongTensor
    p_mask: torch.FloatTensor
    tokens: list


class SquadTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.SQUAD_STYLE_QA

    def __init__(self, name, path_dict):
        super().__init__(name=name, path_dict=path_dict)

    def get_train_examples(self):
        return self.read_squad_examples(path=self.train_path, set_type=PHASE.TRAIN)

    def get_val_examples(self):
        return self.read_squad_examples(path=self.val_path, set_type=PHASE.VAL)

    def get_test_examples(self):
        return self.read_squad_examples(path=self.test_path, set_type=PHASE.TEST)

    @classmethod
    def read_squad_examples(cls, path, set_type):
        with open(path, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        is_training = set_type == PHASE.TRAIN
        examples = []
        for entry in tqdm.tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = Example(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
        return examples


def is_whitespace(c_):
    if c_ == " " or c_ == "\t" or c_ == "\r" or c_ == "\n" or ord(c_) == 0x202F:
        return True
    return False


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)
