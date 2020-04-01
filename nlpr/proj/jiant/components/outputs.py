from dataclasses import dataclass
from typing import Any, Dict

import torch

import nlpr.shared.pycore as pycore


class BaseModelOutput(pycore.ExtendedDataClassMixin):
    pass


@dataclass
class LogitsOutput(BaseModelOutput):
    logits: torch.Tensor
    other: Any = None


@dataclass
class LogitsAndLossOutput(BaseModelOutput):
    logits: torch.Tensor
    loss: torch.Tensor
    other: Any = None


def construct_output_from_dict(struct_dict: Dict):
    keys = sorted(list(struct_dict.keys()))
    if keys == ["logits", "other"]:
        return LogitsOutput.from_dict(struct_dict)
    elif keys == ["logits", "loss", "other"]:
        return LogitsAndLossOutput.from_dict(struct_dict)
    else:
        raise ValueError()
