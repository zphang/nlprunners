import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

import transformers.modeling_bert as modeling_bert
import nlpr.shared.torch_utils as torch_utils
import nlpr.shared.model_resolution as model_resolution

import pyutils.io as io

DEFAULT_ADAPTER_SIZE = 64
DEFAULT_ADAPTER_INITIALIZER_RANGE = 0.0002


@dataclass
class AdapterConfig:
    hidden_act: str = "gelu"
    adapter_size: int = 64
    adapter_initializer_range: float = 0.0002


class Adapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_config: AdapterConfig):
        super(Adapter, self).__init__()
        self.hidden_size = hidden_size
        self.adapter_config = adapter_config

        self.down_project = nn.Linear(
            self.hidden_size,
            self.adapter_config.adapter_size,
        )
        self.activation = modeling_bert.ACT2FN[self.adapter_config.hidden_act] \
            if isinstance(self.adapter_config.hidden_act, str) else self.adapter_config.hidden_act
        self.up_project = nn.Linear(self.adapter_config.adapter_size, self.hidden_size)
        self.init_weights()

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected

    def init_weights(self):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()


class BertOutputWithAdapters(nn.Module):
    def __init__(self, dense, adapter, layer_norm, dropout):
        super(BertOutputWithAdapters, self).__init__()
        self.dense = dense
        self.adapter = adapter
        self.LayerNorm = layer_norm
        self.dropout = dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    @classmethod
    def from_original(cls, old_module, adapter_config: AdapterConfig):
        assert isinstance(old_module, modeling_bert.BertOutput)
        return cls(
            dense=old_module.dense,
            adapter=Adapter(
                hidden_size=old_module.dense.out_features,
                adapter_config=adapter_config,
            ),
            layer_norm=old_module.LayerNorm,
            dropout=old_module.dropout,
        )


class BertSelfOutputWithAdapters(nn.Module):
    def __init__(self, dense, adapter, layer_norm, dropout):
        super(BertSelfOutputWithAdapters, self).__init__()
        self.dense = dense
        self.adapter = adapter
        self.LayerNorm = layer_norm
        self.dropout = dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    @classmethod
    def from_original(cls, old_module, adapter_config: AdapterConfig):
        assert isinstance(old_module, modeling_bert.BertSelfOutput)
        return cls(
            dense=old_module.dense,
            adapter=Adapter(
                hidden_size=old_module.dense.out_features,
                adapter_config=adapter_config,
            ),
            layer_norm=old_module.LayerNorm,
            dropout=old_module.dropout,
        )


def add_adapters(model, adapter_config):
    modified = {}
    for p_name, p_module, c_name, c_module in torch_utils.get_parent_child_module_list(model):
        model_architecture = model_resolution.ModelArchitectures.from_ptt_model(model)
        if model_architecture in [model_resolution.ModelArchitectures.BERT,
                                  model_resolution.ModelArchitectures.ROBERTA]:
            if isinstance(c_module, modeling_bert.BertOutput):
                new_module = BertOutputWithAdapters.from_original(
                    old_module=c_module,
                    adapter_config=adapter_config,
                )
                setattr(p_module, c_name, new_module)
                modified[f"{p_name}.{c_name}"] = new_module
            elif isinstance(c_module, modeling_bert.BertSelfOutput):
                new_module = BertSelfOutputWithAdapters.from_original(
                    old_module=c_module,
                    adapter_config=adapter_config,
                )
                setattr(p_module, c_name, new_module)
                modified[f"{p_name}.{c_name}"] = new_module
        else:
            raise KeyError(model_architecture)
    return modified


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
            weights = F.softmax(self.weights, dim=0)
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
    def from_original(cls, old_module, sub_module_name_list,
                      adapter_config: AdapterConfig, do_weighted_softmax=True):
        assert isinstance(old_module, modeling_bert.BertOutput)
        adapter_dict = {"base": torch_utils.IdentityModule()}
        layer_norm_dict = {"base": old_module.LayerNorm}
        for name in sub_module_name_list:
            adapter_dict[name] = Adapter(
                hidden_size=old_module.dense.out_features,
                adapter_config=adapter_config,
            )
            layer_norm_dict[name] = modeling_bert.BertLayerNorm(
                normalized_shape=old_module.LayerNorm.normalized_shape,
                eps=old_module.LayerNorm.eps,
            )
        weighted_sum = WeightedSum(
            name_list=["base"] + sub_module_name_list,
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
    def from_original(cls, old_module, sub_module_name_list,
                      adapter_config: AdapterConfig,
                      do_weighted_softmax=True,
                      include_base=True):
        assert isinstance(old_module, modeling_bert.BertSelfOutput)
        adapter_dict = {}
        layer_norm_dict = {}
        if include_base:
            adapter_dict["base"] = torch_utils.IdentityModule()
            layer_norm_dict["base"] = old_module.LayerNorm
        for name in sub_module_name_list:
            adapter_dict[name] = Adapter(
                hidden_size=old_module.dense.out_features,
                adapter_config=adapter_config,
            )
            layer_norm_dict[name] = modeling_bert.BertLayerNorm(
                normalized_shape=old_module.LayerNorm.normalized_shape,
                eps=old_module.LayerNorm.eps,
            )
        weighted_sum = WeightedSum(
            name_list=["base"] + sub_module_name_list,
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
                       do_weighted_softmax=True, share_weights=False):
    modified_layers = {}
    for p_name, p_module, c_name, c_module in torch_utils.get_parent_child_module_list(model):
        model_architecture = model_resolution.ModelArchitectures.from_ptt_model(model)
        if model_architecture in [model_resolution.ModelArchitectures.BERT,
                                  model_resolution.ModelArchitectures.ROBERTA]:
            if isinstance(c_module, modeling_bert.BertOutput):
                new_module = BertOutputWithMultiAdapters.from_original(
                    old_module=c_module,
                    sub_module_name_list=sub_module_name_list,
                    adapter_config=adapter_config,
                    do_weighted_softmax=do_weighted_softmax,
                )
                setattr(p_module, c_name, new_module)
                modified_layers[f"{p_name}.{c_name}"] = new_module
            elif isinstance(c_module, modeling_bert.BertSelfOutput):
                new_module = BertSelfOutputWithMultiAdapters.from_original(
                    old_module=c_module,
                    sub_module_name_list=sub_module_name_list,
                    adapter_config=adapter_config,
                    do_weighted_softmax=do_weighted_softmax,
                )
                setattr(p_module, c_name, new_module)
                modified_layers[f"{p_name}.{c_name}"] = new_module
        else:
            raise KeyError(model_architecture)

    if share_weights:
        shared_weighted_sum = None
        for modified_module in modified_layers.values():
            if shared_weighted_sum is None:
                shared_weighted_sum = modified_module.weighted_sum
            else:
                modified_module.weighted_sum = shared_weighted_sum

    return modified_layers


def load_multi_adapter_weights(model, modified_layers: dict, adapter_weights_dict):
    for adapter_set_name, weights_dict in adapter_weights_dict.items():
        model_architecture = model_resolution.ModelArchitectures.from_ptt_model(model)
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


def load_non_adapter_base_weights(model, state_dict):
    curr_state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k in curr_state_dict:
            curr_state_dict[k] = v
    model.load_state_dict(curr_state_dict)


def get_adapter_params(model):
    """
    Gets list of adapter parameters from a model

    :param model: nn.Module
    :return: list
    """
    adapter_params = []
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, (Adapter, modeling_bert.BertLayerNorm)):
            adapter_params += [
                (name + "." + k, v)
                for k, v in sub_module.named_parameters()
            ]
    return adapter_params


def get_multi_adapter_weight_params(model):
    weight_params = []
    for name, param in model.named_parameters():
        if "weighted_sum" in name:
            weight_params.append((name, param))
    return weight_params


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
