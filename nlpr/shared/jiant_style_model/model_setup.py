from dataclasses import dataclass
from typing import Any

import transformers as ptt

import nlpr.shared.model_setup as model_setup
from nlpr.shared.model_setup import ModelArchitectures
import nlpr.shared.jiant_style_model.primary as primary
import nlpr.shared.jiant_style_model.submodels as submodels


def setup_jiant_style_model(model_type, config_path, tokenizer_path, task_dict):
    model_arch = ModelArchitectures.from_model_type(model_type)
    ptt_class_spec = PTT_CLASS_SPEC_DICT[model_arch]
    tokenizer = model_setup.get_tokenizer(
        model_type=model_type,
        tokenizer_class=ptt_class_spec.tokenizer_class,
        tokenizer_path=tokenizer_path,
    )
    ancestor_model = get_ancestor_model(
        ptt_class_spec=ptt_class_spec,
        config_path=config_path,
    )
    submodels_dict = {
        task_name: create_submodel(
            task=task,
            model_arch=model_arch,
            ancestor_model=ancestor_model,
            tokenizer=tokenizer,
        )
        for task_name, task in task_dict.items()
    }
    return primary.JiantStyleModel(
        task_dict=task_dict,
        submodels_dict=submodels_dict,
        tokenizer=tokenizer,
    )


def create_submodel(task, model_arch, ancestor_model, tokenizer) -> submodels.Submodel:



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


def get_ancestor_model(ptt_class_spec, config_path):
    config = ptt_class_spec.config_class.from_json_file(config_path)
    model = ptt_class_spec.model_class(config)
    return model

