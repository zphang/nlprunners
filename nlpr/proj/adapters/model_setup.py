import nlpr.shared.model_resolution as model_resolution
import nlpr.proj.adapters.modeling as adapters


def get_head_parameters(model_arch):
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


def get_adapter_named_parameters(model):
    # Todo: Refactor
    named_parameters = adapters.get_adapter_params(model)
    model_arch = model_resolution.ModelArchitectures.from_ptt_model(model)

    full_named_parameters_dict = dict(model.named_parameters())
    for name in get_head_parameters(model_arch):
        named_parameters.append((name, full_named_parameters_dict[name]))
    return named_parameters
