import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers.modeling_bert as modeling_bert

import nlpr.proj.adapters.modeling as adapters_modeling
import nlpr.shared.torch_utils as torch_utils
import nlpr.shared.model_resolution as model_resolution

import pyutils.datastructures as datastructures
import pyutils.io as io


class WeightedSum(nn.Module):
    def __init__(self, name_list, do_softmax=True):
        super().__init__()
        self.name_list = name_list
        self.do_softmax = do_softmax

        self.num = len(self.name_list)
        self.weights = nn.Parameter(torch.ones(self.num) / self.num)
        self.name2idx = dict(zip(self.name_list, range(self.num)))

    def forward(self, x_dict):
        if self.do_softmax:
            weights = F.softmax(self.weights * 10, dim=0)
        else:
            weights = self.weights
        weighted_sum = torch.stack([
            weights[self.name2idx[k]] * x
            for k, x in x_dict.items()
        ], dim=0).sum(dim=0)
        return weighted_sum


class BertOutputWithMultiAdapters(nn.Module):
    def __init__(self, dense, adapter_dict, layer_norm_dict, weighted_sum, dropout):
        super(BertOutputWithMultiAdapters, self).__init__()
        self.dense = dense
        self.adapter_dict = nn.ModuleDict(adapter_dict)
        self.layer_norm_dict = nn.ModuleDict(layer_norm_dict)
        self.weighted_sum = weighted_sum
        self.dropout = dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states_dict = {}
        for k in self.adapter_dict:
            sub_hidden_states = self.adapter_dict[k](hidden_states)
            sub_hidden_states = self.layer_norm_dict[k](sub_hidden_states + input_tensor)
            hidden_states_dict[k] = sub_hidden_states

        combined_hidden_states = self.weighted_sum(hidden_states_dict)
        return combined_hidden_states

    @classmethod
    def from_original(
        cls,
        old_module,
        sub_module_name_list,
        adapter_config: adapters_modeling.AdapterConfig,
        do_weighted_softmax=True,
        include_base=True,
    ):
        assert isinstance(old_module, modeling_bert.BertOutput)
        adapter_dict = {}
        layer_norm_dict = {}
        if include_base:
            adapter_dict["base"] = torch_utils.IdentityModule()
            layer_norm_dict["base"] = old_module.LayerNorm
        for name in sub_module_name_list:
            adapter_dict[name] = adapters_modeling.Adapter(
                hidden_size=old_module.dense.out_features,
                adapter_config=adapter_config,
            )
            layer_norm_dict[name] = modeling_bert.BertLayerNorm(
                normalized_shape=old_module.LayerNorm.normalized_shape,
                eps=old_module.LayerNorm.eps,
            )
        weighted_sum = WeightedSum(
            name_list=list(adapter_dict.keys()),
            do_softmax=do_weighted_softmax,
        )
        return cls(
            dense=old_module.dense,
            adapter_dict=adapter_dict,
            layer_norm_dict=layer_norm_dict,
            weighted_sum=weighted_sum,
            dropout=old_module.dropout,
        )


class BertSelfOutputWithMultiAdapters(nn.Module):
    def __init__(self, dense, adapter_dict, layer_norm_dict, weighted_sum, dropout):
        super(BertSelfOutputWithMultiAdapters, self).__init__()
        self.dense = dense
        self.adapter_dict = nn.ModuleDict(adapter_dict)
        self.layer_norm_dict = nn.ModuleDict(layer_norm_dict)
        self.weighted_sum = weighted_sum
        self.dropout = dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states_dict = {}
        for k in self.adapter_dict:
            sub_hidden_states = self.adapter_dict[k](hidden_states)
            sub_hidden_states = self.layer_norm_dict[k](sub_hidden_states + input_tensor)
            hidden_states_dict[k] = sub_hidden_states

        combined_hidden_states = self.weighted_sum(hidden_states_dict)
        return combined_hidden_states

    @classmethod
    def from_original(
        cls,
        old_module,
        sub_module_name_list,
        adapter_config: adapters_modeling.AdapterConfig,
        do_weighted_softmax=True,
        include_base=True,
    ):
        assert isinstance(old_module, modeling_bert.BertSelfOutput)
        adapter_dict = {}
        layer_norm_dict = {}
        if include_base:
            adapter_dict["base"] = torch_utils.IdentityModule()
            layer_norm_dict["base"] = old_module.LayerNorm
        for name in sub_module_name_list:
            adapter_dict[name] = adapters_modeling.Adapter(
                hidden_size=old_module.dense.out_features,
                adapter_config=adapter_config,
            )
            layer_norm_dict[name] = modeling_bert.BertLayerNorm(
                normalized_shape=old_module.LayerNorm.normalized_shape,
                eps=old_module.LayerNorm.eps,
            )
        weighted_sum = WeightedSum(
            name_list=list(adapter_dict.keys()),
            do_softmax=do_weighted_softmax,
        )
        return cls(
            dense=old_module.dense,
            adapter_dict=adapter_dict,
            layer_norm_dict=layer_norm_dict,
            weighted_sum=weighted_sum,
            dropout=old_module.dropout,
        )


def add_multi_adapters(model, sub_module_name_list, adapter_config,
                       do_weighted_softmax=True,
                       include_base=True,
                       num_weight_sets: int = 1):
    modified_layers = {}
    model_architecture = model_resolution.ModelArchitectures.from_ptt_model(model)

    for p_name, p_module, c_name, c_module in torch_utils.get_parent_child_module_list(model):
        if model_architecture in [model_resolution.ModelArchitectures.BERT,
                                  model_resolution.ModelArchitectures.ROBERTA]:
            # Drop "roberta." or "bert."
            if isinstance(c_module, modeling_bert.BertOutput):
                p_name = p_name.split(".", 1)[1]
                new_module = BertOutputWithMultiAdapters.from_original(
                    old_module=c_module,
                    sub_module_name_list=sub_module_name_list,
                    adapter_config=adapter_config,
                    do_weighted_softmax=do_weighted_softmax,
                    include_base=include_base,
                )
                setattr(p_module, c_name, new_module)
                modified_layers[f"{p_name}.{c_name}"] = new_module
            elif isinstance(c_module, modeling_bert.BertSelfOutput):
                p_name = p_name.split(".", 1)[1]
                new_module = BertSelfOutputWithMultiAdapters.from_original(
                    old_module=c_module,
                    sub_module_name_list=sub_module_name_list,
                    adapter_config=adapter_config,
                    do_weighted_softmax=do_weighted_softmax,
                    include_base=include_base,
                )
                setattr(p_module, c_name, new_module)
                modified_layers[f"{p_name}.{c_name}"] = new_module
        else:
            raise KeyError(model_architecture)

    if model_architecture in [model_resolution.ModelArchitectures.BERT,
                              model_resolution.ModelArchitectures.ROBERTA]:
        num_per_layer = 2
    else:
        raise KeyError(model_architecture)

    if num_weight_sets == -1:
        num_weight_sets = len(modified_layers) // num_per_layer
    assert len(modified_layers) % (num_per_layer * num_weight_sets) == 0

    layers_sets = datastructures.partition_list(list(modified_layers.items()), num_weight_sets)
    for layer_set in layers_sets:
        shared_weighted_sum = None
        for _, modified_module in layer_set:
            if shared_weighted_sum is None:
                shared_weighted_sum = modified_module.weighted_sum
            else:
                modified_module.weighted_sum = shared_weighted_sum

    return modified_layers


def load_multi_adapter_weights(model, modified_layers: dict, adapter_weights_dict):
    """
    encoder_name_dict = {
        model_resolution.ModelArchitectures.BERT: "bert",
        model_resolution.ModelArchitectures.ROBERTA: "roberta",
    }
    """

    for adapter_set_name, weights_dict in adapter_weights_dict.items():
        model_architecture = model_resolution.ModelArchitectures.from_ptt_model(model)
        # encoder_name = encoder_name_dict[model_architecture]
        if model_architecture in [model_resolution.ModelArchitectures.BERT,
                                  model_resolution.ModelArchitectures.ROBERTA]:
            for name, module in modified_layers.items():
                module_state_dict = module.state_dict()
                module_state_dict[f"layer_norm_dict.{adapter_set_name}.weight"] = \
                    weights_dict[f"{name}.LayerNorm.weight"]
                module_state_dict[f"layer_norm_dict.{adapter_set_name}.bias"] = \
                    weights_dict[f"{name}.LayerNorm.bias"]
                module_state_dict[f"adapter_dict.{adapter_set_name}.down_project.weight"] = \
                    weights_dict[f"{name}.adapter.down_project.weight"]
                module_state_dict[f"adapter_dict.{adapter_set_name}.down_project.bias"] = \
                    weights_dict[f"{name}.adapter.down_project.bias"]
                module_state_dict[f"adapter_dict.{adapter_set_name}.up_project.weight"] = \
                    weights_dict[f"{name}.adapter.up_project.weight"]
                module_state_dict[f"adapter_dict.{adapter_set_name}.up_project.bias"] = \
                    weights_dict[f"{name}.adapter.up_project.bias"]
                module.load_state_dict(module_state_dict)
        else:
            raise KeyError(model_architecture)


def get_multi_adapter_weight_params(model):
    weight_params = []
    for name, param in model.named_parameters():
        if "weighted_sum" in name:
            weight_params.append((name, param))
    return weight_params


def get_multi_adapter_adapter_params_dict(modified_layers):
    adapter_params_dict = {}
    for layer_name, layer in modified_layers.items():
        for adapter_set_name, adapter in layer.adapter_dict.items():
            if adapter_set_name == "base":
                continue
            if adapter_set_name not in adapter_params_dict:
                adapter_params_dict[adapter_set_name] = []
            for name, param in adapter.named_parameters():
                adapter_params_dict[adapter_set_name].append((
                    f"{layer_name}.adapter_dict.{adapter_set_name}.{name}",
                    param,
                ))
    return adapter_params_dict


def get_multi_adapter_weight_dict(modified_layers, simplify_name=True):
    result = {}
    for i, (name, layer) in enumerate(modified_layers.items()):
        if simplify_name:
            name = f"layer_{i:02d}"
        weights = F.softmax(layer.weighted_sum.weights.data.clone(), dim=-1).cpu().numpy()
        names = layer.weighted_sum.name_list
        result[name] = dict(zip(names, weights))
    return result


def load_adapter_weights_dict(path_dict):
    return {
        name: torch.load(path, map_location="cpu")
        for name, path in path_dict.items()
    }


def load_adapter_weights_dict_path(path):
    return load_adapter_weights_dict(io.read_json(path))
