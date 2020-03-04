import numpy as np
import torch
from dataclasses import dataclass
from typing import List

import pyutils.io as io

import transformers as ptt
from nlpr.tasks.lib.templates.shared import (
    Task, TaskTypes, construct_single_input_tokens_and_segment_ids,
)
from ..core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    tag_ids: List[int]

    def tokenize(self, tokenizer):
        tokenized = tokenizer.tokenize(self.text)
        split_text = self.text.split(" ")
        input_flat_stripped = input_flat_strip(split_text)
        flat_stripped, indices = delegate_flat_strip(
            tokens=tokenized,
            tokenizer=tokenizer,
            return_indices=True,
        )
        assert flat_stripped == input_flat_stripped
        positions = map_tags_to_token_position(
            flat_stripped=flat_stripped,
            indices=indices,
            split_text=split_text,
        )
        labels, mask = convert_mapped_tags(
            positions=positions,
            tag_ids=self.tag_ids,
            length=len(tokenized),
        )

        return TokenizedExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.text),
            labels=labels,
            mask=mask,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text: List
    labels: List[int, None]
    mask: List[int]

    def featurize(self, tokenizer, feat_spec):
        unpadded_inputs = construct_single_input_tokens_and_segment_ids(
            input_tokens=self.text,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
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


class CCGTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.TAGGING
    LABELS = range(1363)
    LABEL_BIMAP = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(self.val_path, set_type="val")

    def get_test_examples(self):
        raise NotImplementedError()

    def _create_examples(self, path, set_type):
        tags_to_id = io.read_json(self.path_dict["tags_to_id"])
        examples = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                text, tags = line.strip().split("\t")
                split_tags = tags.split()
                tag_ids = [tags_to_id[tag] for tag in split_tags]
                examples.append(Example(
                    guid="%s-%s" % (set_type, i),
                    text=text,
                    tag_ids=tag_ids,
                ))
        return examples


def map_tags_to_token_position(flat_stripped, indices, split_text):
    char_index = 0
    current_string = flat_stripped
    result = [None] * len(split_text)
    for i, token in enumerate(split_text):
        found_index = current_string.find(token.lower())
        assert found_index != -1
        result[i] = indices[char_index + found_index]
        char_index += found_index + len(token)
        current_string = flat_stripped[char_index:]
    for elem in result:
        assert elem is not None
    return result


def convert_mapped_tags(positions, tag_ids, length):
    labels = [None] * length
    mask = [0] * length
    for pos, tag_id in zip(positions, tag_ids):
        labels[pos] = tag_ids
        mask[pos] = 1
    return labels, mask


def input_flat_strip(tokens):
    return "".join(tokens).lower()


def delegate_flat_strip(tokens, tokenizer, return_indices=False):
    if isinstance(tokenizer, ptt.BertTokenizer):
        return bert_flat_strip(tokens=tokens, return_indices=return_indices)
    elif isinstance(tokenizer, ptt.RobertaTokenizer):
        return roberta_flat_strip(tokens=tokens, return_indices=return_indices)
    elif isinstance(tokenizer, ptt.AlbertTokenizer):
        return albert_flat_strip(tokens=tokens, return_indices=return_indices)
    else:
        raise KeyError(type(tokenizer))


def bert_flat_strip(tokens, return_indices=False):
    ls = []
    indices = []
    for token in tokens:
        if token.startswith("##"):
            token = token.replace("##", "")
        else:
            pass
        ls.append(token)
        indices += [len(indices)] * len(token)
    string =  "".join(ls).lower()
    if return_indices:
        return string, indices
    else:
        return string


def roberta_flat_strip(tokens, return_indices=False):
    ls = []
    indices = []
    for token in tokens:
        if token.startswith("Ġ"):
            token = token.replace("Ġ", "")
        else:
            pass
        ls.append(token)
        indices += [len(indices)] * len(token)
    string = "".join(ls).lower()
    if return_indices:
        return string, indices
    else:
        return string


def albert_flat_strip(tokens, return_indices=False):
    ls = []
    indices = []
    for token in tokens:
        token = token.replace('"', "``")
        if token.startswith("▁"):
            token = token[1:]
        else:
            pass
        ls.append(token)
        indices += [len(indices)] * len(token)
    string = "".join(ls).lower()
    if return_indices:
        return string, indices
    else:
        return string
