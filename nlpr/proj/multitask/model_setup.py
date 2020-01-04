import pyutils

import nlpr.shared.model_resolution as model_resolution
import nlpr.shared.model_setup as model_setup
from nlpr.shared.model_resolution import ModelArchitectures
import nlpr.proj.multitask.modeling as multitask_modeling


def setup_multitask_ptt_model(model_type, config_path, tokenizer_path, task_dict):
    model_arch = ModelArchitectures.from_model_type(model_type)
    assert ModelArchitectures.is_ptt_model_arch(model_arch)

    # 1. Retrieve class specs
    model_class_spec_dict = {}
    for task_name, task in task_dict.items():
        model_class_spec_dict[task_name] = model_resolution.resolve_model_setup_classes(
            model_type=model_type,
            task_type=task.TASK_TYPE,
        )

    # 2. Get tokenizer
    tokenizer_class_list = [
        model_class_spec.tokenizer_class
        for model_class_spec in model_class_spec_dict.values()
    ]
    tokenizer_class_list = list(set(tokenizer_class_list))
    tokenizer = model_setup.get_tokenizer(
        model_type=model_type,
        tokenizer_class=pyutils.take_one(tokenizer_class_list),
        tokenizer_path=tokenizer_path,
    )

    # 3. Get model
    shared_ptt_encoder = None
    model_dict = {}
    for task_name, task in task_dict.items():
        task_model = model_setup.get_model(
            model_class_spec=model_class_spec_dict[task_name],
            config_path=config_path,
            task=task,
        )
        encoder = get_ptt_encoder(task_model)
        if shared_ptt_encoder is None:
            shared_ptt_encoder = encoder
        else:
            set_ptt_encoder(task_model, shared_ptt_encoder)
        model_dict[task_name] = task_model

    multitask_model = multitask_modeling.MultiTaskModel(
        model_dict=model_dict,
        shared_ptt_encoder=shared_ptt_encoder,
    )

    return model_setup.ModelWrapper(
        model=multitask_model,
        tokenizer=tokenizer,
    )


def _get_ptt_encoder_attr(ptt_model):
    model_arch = ModelArchitectures.from_ptt_model(ptt_model)
    # Will probably need to refactor this out later
    if model_arch == ModelArchitectures.BERT:
        return "bert"
    elif model_arch == ModelArchitectures.XLNET:
        return "transformer"
    elif model_arch == ModelArchitectures.XLM:
        return "transformer"
    elif model_arch == ModelArchitectures.ROBERTA:
        return "roberta"
    else:
        raise KeyError(model_arch)


def get_ptt_encoder(ptt_model):
    return getattr(ptt_model, _get_ptt_encoder_attr(ptt_model))


def set_ptt_encoder(ptt_model, encoder):
    return setattr(ptt_model, _get_ptt_encoder_attr(ptt_model), encoder)

