import nlpr.shared.model_resolution as model_resolution
import nlpr.proj.adapters.modeling as adapters


def get_head_parameter_names(model):
    model_arch = model_resolution.ModelArchitectures.from_ptt_model(model)
    if model_arch == model_resolution.ModelArchitectures.BERT:
        return [
            "bert.pooler.dense.weight",
            "bert.pooler.dense.bias",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
            "classifier.weight",
            "classifier.bias",
        ]
    elif model_arch == model_resolution.ModelArchitectures.ROBERTA:
        return [
            "roberta.pooler.dense.weight",
            "roberta.pooler.dense.bias",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
            "classifier.dense.weight",
            "classifier.dense.bias",
            "classifier.dense.weight",
            "classifier.dense.bias",
        ]
    else:
        raise KeyError()


def get_head_named_parameters(model):
    head_parameter_names = get_head_parameter_names(model)
    full_named_parameters_dict = dict(model.named_parameters())
    return [
        (param_name, full_named_parameters_dict[param_name])
        for param_name in head_parameter_names
    ]


def get_adapter_named_parameters(model):
    # Todo: Refactor
    named_parameters = adapters.get_adapter_params(model)
    return named_parameters + get_head_named_parameters(model)
