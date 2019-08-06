from dataclasses import dataclass
from enum import Enum

import pytorch_transformers as ptt

from nlpr.tasks.core import FeaturizationSpec
from nlpr.tasks.lib.shared import TaskTypes


class ModelArchitectures(Enum):
    BERT = 1
    XLNET = 2
    XLM = 3

    @classmethod
    def from_model_type(cls, model_type):
        if model_type.startswith("bert-"):
            return cls.BERT
        elif model_type.startswith("xlnet-"):
            return cls.XLNET
        elif model_type.startswith("xlm-"):
            return cls.XLM
        else:
            raise KeyError(model_type)

    @classmethod
    def from_ptt_model(cls, ptt_model):
        if isinstance(ptt_model, ptt.BertPreTrainedModel):
            return cls.BERT
        elif isinstance(ptt_model, ptt.XLNetPreTrainedModel):
            return cls.XLNET
        elif isinstance(ptt_model, ptt.XLMPreTrainedModel):
            return cls.XLM
        else:
            raise KeyError(str(ptt_model))


@dataclass
class ModelClassSpec:
    config_class: type
    tokenizer_class: type
    model_class: type


def build_featurization_spec(model_type, max_seq_length):
    model_arch = ModelArchitectures.from_model_type(model_type)
    if model_arch == ModelArchitectures.BERT:
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=0,
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=1,
        )
    elif model_arch == ModelArchitectures.XLNET:
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=True,
            pad_on_left=True,
            cls_token_segment_id=2,
            pad_token_segment_id=4,
            pad_token_id=0,
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=1,
        )
    elif model_arch == ModelArchitectures.XLM:
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=0,
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=1,
        )
    else:
        raise KeyError(model_arch)


MODEL_CLASS_DICT = {
    ModelArchitectures.BERT: {
        TaskTypes.CLASSIFICATION: ptt.BertForSequenceClassification,
        TaskTypes.REGRESSION: None,  # todo, regression
    },
    ModelArchitectures.XLNET: {
        TaskTypes.CLASSIFICATION: ptt.XLNetForSequenceClassification,
        TaskTypes.REGRESSION: None,  # todo, regression
    },
    ModelArchitectures.XLM: {
        TaskTypes.CLASSIFICATION: ptt.XLMForSequenceClassification,
        TaskTypes.REGRESSION: None,  # todo, regression
    },
}


def resolve_model_setup_classes(model_type, task_type):
    model_arch = ModelArchitectures.from_model_type(model_type)
    if model_arch == ModelArchitectures.BERT:
        model_class_spec = ModelClassSpec(
            config_class=ptt.BertConfig,
            tokenizer_class=ptt.BertTokenizer,
            # TODO: resolve correct model
            model_class=MODEL_CLASS_DICT[ModelArchitectures.BERT][task_type],
        )
    elif model_arch == ModelArchitectures.XLNET:
        model_class_spec = ModelClassSpec(
            config_class=ptt.XLNetConfig,
            tokenizer_class=ptt.XLNetTokenizer,
            # TODO: resolve correct model
            model_class=MODEL_CLASS_DICT[ModelArchitectures.XLNET][task_type],
        )
    elif model_arch == ModelArchitectures.XLM:
        model_class_spec = ModelClassSpec(
            config_class=ptt.XLMConfig,
            tokenizer_class=ptt.XLMTokenizer,
            # TODO: resolve correct model
            model_class=MODEL_CLASS_DICT[ModelArchitectures.XLM][task_type],
        )
    else:
        raise KeyError(model_arch)
    return model_class_spec
