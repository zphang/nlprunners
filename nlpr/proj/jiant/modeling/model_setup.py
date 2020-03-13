from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import transformers as ptt

import pyutils.strings as strings

import nlpr.shared.model_setup as model_setup
import nlpr.proj.jiant.modeling.primary as primary
import nlpr.proj.jiant.modeling.submodels as submodels
import nlpr.proj.jiant.modeling.heads as heads
from nlpr.shared.model_setup import ModelArchitectures
from nlpr.tasks import TaskTypes


def setup_jiant_style_model(model_type, model_config_path, tokenizer_path, task_dict):
    model_arch = ModelArchitectures.from_model_type(model_type)
    ptt_class_spec = PTT_CLASS_SPEC_DICT[model_arch]
    tokenizer = model_setup.get_tokenizer(
        model_type=model_type,
        tokenizer_class=ptt_class_spec.tokenizer_class,
        tokenizer_path=tokenizer_path,
    )
    ancestor_model = get_ancestor_model(
        ptt_class_spec=ptt_class_spec,
        model_config_path=model_config_path,
    )
    encoder = get_encoder(model_arch=model_arch, ancestor_model=ancestor_model)
    submodels_dict = {
        task_name: create_submodel(
            task=task,
            model_arch=model_arch,
            encoder=encoder,
        )
        for task_name, task in task_dict.items()
    }
    return primary.JiantStyleModel(
        task_dict=task_dict,
        encoder=encoder,
        submodels_dict=submodels_dict,
        tokenizer=tokenizer,
    )


def setup_jiant_style_model_single(model_type, model_config_path, tokenizer_path, task):
    return setup_jiant_style_model(
        model_type=model_type,
        model_config_path=model_config_path,
        tokenizer_path=tokenizer_path,
        task_dict={
            task.name: task,
        }
    )


def delegate_load_from_path(jiant_model: primary.JiantStyleModel, weights_path: str, load_mode: str):
    weights_dict = torch.load(weights_path)
    return delegate_load(
        jiant_model=jiant_model,
        weights_dict=weights_dict,
        load_mode=load_mode,
    )


def delegate_load(jiant_model, weights_dict: dict, load_mode: str):
    if load_mode == "from_ptt":
        return load_encoder_from_ptt_weights(
            encoder=jiant_model.encoder,
            weights_dict=weights_dict,
        )
    elif load_mode == "from_ptt_with_mlm":
        remainder = load_encoder_from_ptt_weights(
            encoder=jiant_model.encoder,
            weights_dict=weights_dict,
            return_remainder=True,
        )
        x = load_lm_heads_from_ptt_weights(
            jiant_model=jiant_model,
            weights_dict=remainder,
        )
        return
    elif load_mode == "all":
        jiant_model.load_state_dict(weights_dict)
    elif load_mode == "partial_weights":
        return load_partial_heads(
            jiant_model=jiant_model,
            weights_dict=weights_dict,
            allow_missing_head_weights=True,
        )
    elif load_mode == "partial_heads":
        return load_partial_heads(
            jiant_model=jiant_model,
            weights_dict=weights_dict,
            allow_missing_head_model=True,
        )
    elif load_mode == "partial":
        return load_partial_heads(
            jiant_model=jiant_model,
            weights_dict=weights_dict,
            allow_missing_head_weights=True,
            allow_missing_head_model=True,
        )
    else:
        raise KeyError(load_mode)


def load_encoder_from_ptt_weights(encoder: nn.Module,
                                  weights_dict: dict,
                                  with_lm=False, return_remainder=False):
    remainder_weights_dict = {}
    load_weights_dict = {}
    model_arch = get_model_arch_from_encoder(encoder=encoder)
    encoder_prefix = MODEL_PREFIX[model_arch] + "."
    # Encoder
    for k, v in weights_dict.items():
        if k.startswith(encoder_prefix):
            load_weights_dict[strings.remove_prefix(k, encoder_prefix)] = v
        else:
            remainder_weights_dict[k] = v
    encoder.load_state_dict(load_weights_dict)
    if return_remainder:
        return remainder_weights_dict


def load_lm_heads_from_ptt_weights(jiant_model, weights_dict):
    model_arch = get_model_arch_from_jiant_model(jiant_model=jiant_model)
    if model_arch == ModelArchitectures.BERT:
        mlm_weights_dict = {
            'bias': "cls.predictions.bias",
            'dense.weight': "cls.predictions.transform.dense.weight",
            'dense.bias': "cls.predictions.transform.dense.bias",
            'LayerNorm.weight': "cls.predictions.transform.LayerNorm.weight",
            'LayerNorm.bias': "cls.predictions.transform.LayerNorm.bias",
            'decoder.weight': "cls.predictions.decoder.weight",
            # 'decoder.bias' <-- linked directly to bias
        }
    elif model_arch == ModelArchitectures.ROBERTA:
        mlm_weights_dict = {
            strings.remove_prefix(k, "lm_head."): v
            for k, v in weights_dict.items()
        }
        mlm_weights_dict["decoder.bias"] = mlm_weights_dict["bias"]
    elif model_arch == ModelArchitectures.ALBERT:
        mlm_weights_dict = {
            strings.remove_prefix(k, "predictions."): v
            for k, v in weights_dict.items()
        }
    else:
        raise KeyError(model_arch)
    missed = set()
    for submodel_name, submodel in jiant_model.submodels_dict.items():
        if not isinstance(submodel, submodels.MLMModel):
            continue
        mismatch = submodel.mlm_head.load_state_dict(mlm_weights_dict)
        assert not mismatch.missing_keys
        missed.update(mismatch.unexpected_keys)
    return list(missed)


def load_partial_heads(jiant_model, weights_dict,
                       allow_missing_head_weights=False,
                       allow_missing_head_model=False):
    mismatch = jiant_model.load_state_dict(weights_dict, strict=False)
    result = {}
    if mismatch.missing_keys:
        assert allow_missing_head_weights
        missing_head_weights = set()
        for k in mismatch.missing_keys:
            missing_head_weights.add(k.split(".")[1])
        result["missing_head_weights"] = list(missing_head_weights)
    if mismatch.unexpected_keys:
        assert allow_missing_head_model
        missing_heads_model = set()
        for k in mismatch.unexpected_keys:
            missing_heads_model.add(k.split(".")[1])
        result["missing_heads_model"] = list(missing_heads_model)
    return result


def create_submodel(task, model_arch, encoder) -> submodels.Submodel:
    if task.TASK_TYPE == TaskTypes.CLASSIFICATION:
        classification_head = heads.ClassificationHead(
            hidden_size=encoder.config.hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
            num_labels=len(task.LABELS),
        )
        submodel = submodels.ClassificationModel(
            encoder=encoder,
            classification_head=classification_head,
        )
    elif task.TASK_TYPE == TaskTypes.REGRESSION:
        regression_head = heads.RegressionHead(
            hidden_size=encoder.config.hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
        )
        submodel = submodels.RegressionModel(
            encoder=encoder,
            regression_head=regression_head,
        )
    elif task.TASK_TYPE == TaskTypes.MULTIPLE_CHOICE:
        choice_scoring_head = heads.RegressionHead(
            hidden_size=encoder.config.hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
        )
        submodel = submodels.MultipleChoiceModel(
            encoder=encoder,
            num_choices=task.NUM_CHOICES,
            choice_scoring_head=choice_scoring_head,
        )
    elif task.TASK_TYPE == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        span_comparison_head = heads.SpanComparisonHead(
            hidden_size=encoder.config.hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
            num_spans=task.num_spans,
            num_labels=len(task.LABELS),
        )
        submodel = submodels.SpanComparisonModel(
            encoder=encoder,
            span_comparison_head=span_comparison_head,
        )
    elif task.TASK_TYPE == TaskTypes.TAGGING:
        token_classification_head = heads.TokenClassificationHead(
            hidden_size=encoder.config.hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
            num_labels=len(task.LABELS),
        )
        submodel = submodels.TokenClassificationModel(
            encoder=encoder,
            token_classification_head=token_classification_head,
        )
    elif task.TASK_TYPE == TaskTypes.SQUAD_STYLE_QA:
        qa_head = heads.QAHead(
            hidden_size=encoder.config.hidden_size,
        )
        submodel = submodels.QAModel(
            encoder=encoder,
            qa_head=qa_head,
        )
    elif task.TASK_TYPE == TaskTypes.MASKED_LANGUAGE_MODELING:
        if model_arch == ModelArchitectures.BERT:
            mlm_head = heads.BertMLMHead(
                hidden_size=encoder.config.hidden_size,
                vocab_size=encoder.config.vocab_size,
                layer_norm_eps=encoder.config.layer_norm_eps,
                hidden_act=encoder.config.hidden_act,
            )
        elif model_arch == ModelArchitectures.ROBERTA:
            mlm_head = heads.RobertaMLMHead(
                hidden_size=encoder.config.hidden_size,
                vocab_size=encoder.config.vocab_size,
                layer_norm_eps=encoder.config.layer_norm_eps,
            )
        elif model_arch == ModelArchitectures.ALBERT:
            mlm_head = heads.AlbertMLMHead(
                hidden_size=encoder.config.hidden_size,
                embedding_size=encoder.config.embedding_size,
                vocab_size=encoder.config.vocab_size,
                hidden_act=encoder.config.hidden_act,
            )
        else:
            raise KeyError(model_arch)
        submodel = submodels.MLMModel(
            encoder=encoder,
            mlm_head=mlm_head,
        )
    else:
        raise KeyError(task.TASK_TYPE)
    return submodel


def get_encoder(model_arch, ancestor_model):
    if model_arch == ModelArchitectures.BERT:
        return ancestor_model.bert
    elif model_arch == ModelArchitectures.ROBERTA:
        return ancestor_model.roberta
    elif model_arch == ModelArchitectures.ALBERT:
        return ancestor_model.albert


@dataclass
class PttClassSpec:
    config_class: Any
    tokenizer_class: Any
    model_class: Any


PTT_CLASS_SPEC_DICT = {
    ModelArchitectures.BERT: PttClassSpec(
        config_class=ptt.BertConfig,
        tokenizer_class=ptt.BertTokenizer,
        model_class=ptt.BertForPreTraining,
    ),
    ModelArchitectures.ROBERTA: PttClassSpec(
        config_class=ptt.RobertaConfig,
        tokenizer_class=ptt.RobertaTokenizer,
        model_class=ptt.RobertaForMaskedLM,
    ),
    ModelArchitectures.ALBERT: PttClassSpec(
        config_class=ptt.AlbertConfig,
        tokenizer_class=ptt.AlbertTokenizer,
        model_class=ptt.AlbertForMaskedLM,
    ),
}


def get_model_arch_from_encoder(encoder: nn.Module) -> ModelArchitectures:
    if type(encoder) is ptt.BertModel:
        return ModelArchitectures.BERT
    elif type(encoder) is ptt.RobertaModel:
        return ModelArchitectures.ROBERTA
    elif type(encoder) is ptt.AlbertModel:
        return ModelArchitectures.ALBERT
    else:
        raise KeyError(type(encoder))


def get_model_arch_from_jiant_model(jiant_model: nn.Module) -> ModelArchitectures:
    return get_model_arch_from_encoder(encoder=jiant_model.encoder)


MODEL_PREFIX = {
    ModelArchitectures.BERT: "bert",
    ModelArchitectures.ROBERTA: "roberta",
    ModelArchitectures.ALBERT: "albert",
}


def get_ancestor_model(ptt_class_spec, model_config_path):
    config = ptt_class_spec.config_class.from_json_file(model_config_path)
    model = ptt_class_spec.model_class(config)
    return model
