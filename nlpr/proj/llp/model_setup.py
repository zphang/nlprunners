import torch

import nlpr.shared.model_setup as shared_model_setup
import nlpr.shared.model_resolution as shared_model_resolution
import nlpr.proj.llp.modeling as llp_modeling


def setup_model(model_type, task, llp_embedding_dim,
                config_path, tokenizer_path):
    model_class_spec = shared_model_resolution.resolve_model_setup_classes(
        model_type=model_type,
        task_type=task.TASK_TYPE,
    )
    base_model_wrapper = shared_model_setup.simple_model_setup(
        model_type=model_type,
        model_class_spec=model_class_spec,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        task=task,
    )
    llp_model = llp_modeling.LlpModel(
        ptt_model=base_model_wrapper.model,
        embedding_dim=llp_embedding_dim,
    )
    model_wrapper = shared_model_setup.ModelWrapper(
        model=llp_model,
        tokenizer=base_model_wrapper.tokenizer
    )
    return model_wrapper


def load_model(model: llp_modeling.LlpModel, state_dict, model_load_mode):
    # todo: port to constant
    if model_load_mode == "ptt_only":
        shared_model_setup.safe_load_model(
            model=model.ptt_model,
            state_dict=state_dict,
        )
    elif model_load_mode == "all":
        model.load_state_dict(state_dict)
    elif model_load_mode == "no_load":
        pass
    else:
        raise KeyError(model_load_mode)


def load_model_path(model, model_path, model_load_mode):
    if model_load_mode == "no_load":
        return
    load_model(
        model=model,
        state_dict=torch.load(model_path),
        model_load_mode=model_load_mode,
    )
