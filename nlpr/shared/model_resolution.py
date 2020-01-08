from dataclasses import dataclass
from enum import Enum

import transformers as ptt
import transformers.modeling_albert

from nlpr.tasks.core import FeaturizationSpec
from nlpr.tasks.lib.shared import TaskTypes
import nlpr.shared.modeling.models as models
import nlpr.shared.modeling.glove_lstm as glove_lstm_modeling


class ModelArchitectures(Enum):
    BERT = 1
    XLNET = 2
    XLM = 3
    ROBERTA = 4
    GLOVE_LSTM = 5
    ALBERT = 6

    @classmethod
    def from_model_type(cls, model_type):
        if model_type.startswith("bert-"):
            return cls.BERT
        elif model_type.startswith("xlnet-"):
            return cls.XLNET
        elif model_type.startswith("xlm-"):
            return cls.XLM
        elif model_type.startswith("roberta-"):
            return cls.ROBERTA
        elif model_type.startswith("albert-"):
            return cls.ALBERT
        elif model_type == "glove_lstm":
            return cls.GLOVE_LSTM
        else:
            raise KeyError(model_type)

    @classmethod
    def from_ptt_model(cls, ptt_model):
        if isinstance(ptt_model, ptt.BertPreTrainedModel) \
                and ptt_model.__class__.__name__.startswith("Bert"):
            return cls.BERT
        elif isinstance(ptt_model, ptt.XLNetPreTrainedModel):
            return cls.XLNET
        elif isinstance(ptt_model, ptt.XLMPreTrainedModel):
            return cls.XLM
        elif isinstance(ptt_model, ptt.BertPreTrainedModel) \
                and ptt_model.__class__.__name__.startswith("Robert"):
            return cls.ROBERTA
        elif isinstance(ptt_model, glove_lstm_modeling.GloveLSTMModel):
            return cls.GLOVE_LSTM
        elif isinstance(ptt_model, transformers.modeling_albert.AlbertPreTrainedModel):
            return cls.ALBERT
        else:
            raise KeyError(str(ptt_model))

    @classmethod
    def is_ptt_model_arch(cls, model_arch):
        return model_arch in [
            cls.BERT,
            cls.XLNET,
            cls.XLM,
            cls.ROBERTA,
            cls.ALBERT,
        ]


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
            sep_token_extra=False,
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
            sep_token_extra=False,
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
            sequence_b_segment_id=0,  # RoBERTa has no token_type_ids
            sep_token_extra=False,
        )
    elif model_arch == ModelArchitectures.ROBERTA:
        # RoBERTa is weird
        # token 0 = '<s>' which is the cls_token
        # token 1 = '</s>' which is the sep_token
        # Also two '</s>'s are used between sentences. Yes, not '</s><s>'.
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=1,  # Roberta uses pad_token_id = 1
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=0,  # RoBERTa has no token_type_ids
            sep_token_extra=True,
        )
    elif model_arch == ModelArchitectures.GLOVE_LSTM:
        return glove_lstm_modeling.GloVeEmbeddings.get_feat_spec(
            max_seq_length=max_seq_length,
        )
    elif model_arch == ModelArchitectures.ALBERT:
        #
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,   # ?
            pad_on_left=False,  # ok
            cls_token_segment_id=0,  # ok
            pad_token_segment_id=0,  # ok
            pad_token_id=0,  # I think?
            pad_token_mask_id=0,  # I think?
            sequence_a_segment_id=0,   # I think?
            sequence_b_segment_id=1,   # I think?
            sep_token_extra=False,
        )
    else:
        raise KeyError(model_arch)


MODEL_CLASS_DICT = {
    ModelArchitectures.BERT: {
        TaskTypes.CLASSIFICATION: ptt.BertForSequenceClassification,
        TaskTypes.REGRESSION: models.BertForSequenceRegression,  # ptt is weird
        TaskTypes.SPAN_COMPARISON_CLASSIFICATION: models.BertForSpanComparisonClassification,
        TaskTypes.MULTIPLE_CHOICE: ptt.BertForMultipleChoice,
    },
    ModelArchitectures.XLNET: {
        TaskTypes.CLASSIFICATION: ptt.XLNetForSequenceClassification,
        TaskTypes.REGRESSION: None,  # ptt is weird
    },
    ModelArchitectures.XLM: {
        TaskTypes.CLASSIFICATION: ptt.XLMForSequenceClassification,
        TaskTypes.REGRESSION: None,  # ptt is weird
    },
    ModelArchitectures.ROBERTA: {
        TaskTypes.CLASSIFICATION: ptt.RobertaForSequenceClassification,
        TaskTypes.REGRESSION: ptt.RobertaForSequenceClassification,  # ptt is weird
        TaskTypes.MULTIPLE_CHOICE: ptt.RobertaForMultipleChoice,
    },
    ModelArchitectures.GLOVE_LSTM: {
        TaskTypes.CLASSIFICATION: glove_lstm_modeling.GloveLSTMForSequenceClassification,
        TaskTypes.REGRESSION: glove_lstm_modeling.GloveLSTMForSequenceRegression,
    },
    ModelArchitectures.ALBERT: {
        TaskTypes.CLASSIFICATION: ptt.AlbertForSequenceClassification,
        TaskTypes.REGRESSION: ptt.AlbertForSequenceClassification,  # ptt is weird
    },
}


# This assumes that there is a single encoder module per model
# It is very possible that this abstraction will be broken in the future
MODEL_ENCODER_PREFIX_DICT = {
    ModelArchitectures.BERT: "bert",
    ModelArchitectures.XLNET: "transformer",
    ModelArchitectures.XLM: "transformer",
    ModelArchitectures.ROBERTA: "roberta",
    ModelArchitectures.ALBERT: "albert",
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
    elif model_arch == ModelArchitectures.ROBERTA:
        model_class_spec = ModelClassSpec(
            config_class=ptt.RobertaConfig,
            tokenizer_class=ptt.RobertaTokenizer,
            # TODO: resolve correct model
            model_class=MODEL_CLASS_DICT[ModelArchitectures.ROBERTA][task_type],
        )
    elif model_arch == ModelArchitectures.GLOVE_LSTM:
        model_class_spec = ModelClassSpec(
            config_class=type(None),
            tokenizer_class=glove_lstm_modeling.GloVeEmbeddings,
            # TODO: resolve correct model
            model_class=MODEL_CLASS_DICT[ModelArchitectures.GLOVE_LSTM][task_type],
        )
    elif model_arch == ModelArchitectures.ALBERT:
        model_class_spec = ModelClassSpec(
            config_class=ptt.AlbertConfig,
            tokenizer_class=ptt.AlbertTokenizer,
            # TODO: resolve correct model
            model_class=MODEL_CLASS_DICT[ModelArchitectures.ALBERT][task_type],
        )
    else:
        raise KeyError(model_arch)
    return model_class_spec
