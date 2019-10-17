import torch.nn as nn

from dataclasses import dataclass

import transformers.modeling_bert as modeling_bert
import nlpr.shared.torch_utils as torch_utils
import nlpr.shared.model_resolution as model_resolution

DEFAULT_ADAPTER_HIDDEN_ACT = "gelu"
DEFAULT_ADAPTER_SIZE = 64
DEFAULT_ADAPTER_INITIALIZER_RANGE = 0.0002


@dataclass
class AdapterConfig:
    hidden_act: str = DEFAULT_ADAPTER_HIDDEN_ACT
    adapter_size: int = DEFAULT_ADAPTER_SIZE
    adapter_initializer_range: float = DEFAULT_ADAPTER_INITIALIZER_RANGE


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
