from dataclasses import dataclass

import torch

from nlpr.shared.pycore import ExtendedDataClassMixin


class IDS:
    UNK = 100
    CLS = 101
    SEP = 102
    MASK = 103


@dataclass
class FeaturizationSpec:
    max_seq_length: int
    cls_token_at_end: bool
    pad_on_left: bool
    cls_token_segment_id: int
    pad_token_segment_id: int
    pad_token_id: int
    pad_token_mask_id: int
    sequence_a_segment_id: int
    sequence_b_segment_id: int
    sep_token_extra: bool


class BatchMixin(ExtendedDataClassMixin):
    def to(self, device):
        return self.__class__(**{
            k: self._val_to_device(v, device)
            for k, v in self.asdict().items()
        })

    @classmethod
    def _val_to_device(cls, v, device):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        else:
            return v

    def __len__(self):
        return len(getattr(self, self.fields[0]))


class BaseExample(ExtendedDataClassMixin):
    def tokenize(self, tokenizer):
        raise NotImplementedError


class BaseTokenizedExample(ExtendedDataClassMixin):
    def featurize(self, tokenizer, feat_spec: FeaturizationSpec):
        raise NotImplementedError


class BaseDataRow(ExtendedDataClassMixin):
    pass


class BaseBatch(BatchMixin, ExtendedDataClassMixin):
    @classmethod
    def from_data_rows(cls, data_row_ls):
        raise NotImplementedError


class BiMap:
    def __init__(self, a, b):
        self.a = {}
        self.b = {}
        for i, j in zip(a, b):
            self.a[i] = j
            self.b[j] = i
        assert len(self.a) == len(self.b) == len(a) == len(b)


def labels_to_bimap(labels):
    return BiMap(a=labels, b=list(range(len(labels))))
