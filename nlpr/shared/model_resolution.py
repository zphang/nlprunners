from dataclasses import dataclass

import pytorch_transformers

from nlpr.tasks.core import FeaturizationSpec
from nlpr.tasks.lib.shared import TaskTypes


@dataclass
class ModelClassSpec:
    config_class: type
    tokenizer_class: type
    model_class: type


def build_featurization_spec(model_type, max_seq_length):
    if model_type.startswith("bert-"):
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
    elif model_type.startswith("xlnet-"):
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
    elif model_type.startswith("xlm-"):
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
        raise KeyError(model_type)


MODEL_CLASS_DICT = {
    "bert": {
        TaskTypes.CLASSIFICATION: pytorch_transformers.BertForSequenceClassification,
        TaskTypes.REGRESSION: None,  # todo, regression
    },
    "xlnet": {
        TaskTypes.CLASSIFICATION: pytorch_transformers.XLNetForSequenceClassification,
        TaskTypes.REGRESSION: None,  # todo, regression
    },
    "xlm": {
        TaskTypes.CLASSIFICATION: pytorch_transformers.XLMForSequenceClassification,
        TaskTypes.REGRESSION: None,  # todo, regression
    },
}


def resolve_model_setup_classes(model_type, task_type):
    if model_type.startswith("bert-"):
        model_class_spec = ModelClassSpec(
            config_class=pytorch_transformers.BertConfig,
            tokenizer_class=pytorch_transformers.BertTokenizer,
            # TODO: resolve correct model
            model_class=MODEL_CLASS_DICT["bert"][task_type],
        )
    elif model_type.startswith("xlnet-"):
        model_class_spec = ModelClassSpec(
            config_class=pytorch_transformers.XLNetConfig,
            tokenizer_class=pytorch_transformers.XLNetTokenizer,
            # TODO: resolve correct model
            model_class=MODEL_CLASS_DICT["xlnet"][task_type],
        )
    elif model_type.startswith("xlm-"):
        model_class_spec = ModelClassSpec(
            config_class=pytorch_transformers.XLMConfig,
            tokenizer_class=pytorch_transformers.XLMTokenizer,
            # TODO: resolve correct model
            model_class=MODEL_CLASS_DICT["xlm"][task_type],
        )
    else:
        raise KeyError(model_type)
    return model_class_spec
