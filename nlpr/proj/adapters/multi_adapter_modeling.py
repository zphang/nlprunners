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
    def __init__(self, name_list, mode="softmax"):
        super().__init__()
        self.name_list = name_list
        self.mode = mode

        self.num = len(self.name_list)

        if mode == "softmax":
            init_values = torch.ones(self.num) / self.num
        elif mode == "gates":
            init_values = torch.zeros(self.num)
        elif mode == "weights":
            init_values = torch.ones(self.num)
        else:
            raise KeyError(self.mode)

        self.weights = nn.Parameter(init_values)
        self.name2idx = dict(zip(self.name_list, range(self.num)))

    def forward(self, x_dict):
        stacked_x = torch.stack([
            x_dict[k]
            for k in self.name_list
        ], dim=-2)
        weights = self.compute_weights().view(1, 1, -1, 1)
        weighted_sum = (stacked_x * weights).sum(-2)
        return weighted_sum

    def compute_weights(self):
        if self.mode == "softmax":
            weights = F.softmax(self.weights, dim=0)
        elif self.mode == "gates":
            weights = F.sigmoid(self.weights)
        elif self.mode == "weights":
            weights = self.weights
        else:
            raise KeyError(self.mode)
        return weights


class MultiAdapter(nn.Module):
    def __init__(self, weighted_sum, adapter_dict, layer_norm_dict,
                 sub_module_name_list=None):
        super(MultiAdapter, self).__init__()
        self.weighted_sum = weighted_sum
        self.adapter_dict = nn.ModuleDict(adapter_dict)
        self.layer_norm_dict = nn.ModuleDict(layer_norm_dict)
        if sub_module_name_list is not None:
            self.sub_module_name_list = sub_module_name_list
        else:
            self.sub_module_name_list = list(adapter_dict.keys())

    def forward(self, hidden_states, input_tensor):
        hidden_states_dict = {}
        for k in self.sub_module_name_list:
            sub_hidden_states = self.adapter_dict[k](hidden_states)
            sub_hidden_states = self.layer_norm_dict[k](sub_hidden_states + input_tensor)
            hidden_states_dict[k] = sub_hidden_states
        combined_hidden_states = self.weighted_sum(hidden_states_dict)
        return combined_hidden_states

    @classmethod
    def create(cls,
               old_parent_module,
               sub_module_name_list,
               adapter_config: adapters_modeling.AdapterConfig,
               mode="softmax",
               include_base=True):
        adapter_dict = {}
        layer_norm_dict = {}
        sub_module_name_list = sub_module_name_list.copy()
        if include_base:
            adapter_dict["base"] = torch_utils.IdentityModule()
            layer_norm_dict["base"] = old_parent_module.LayerNorm
            sub_module_name_list.append("base")
        for name in sub_module_name_list:
            adapter_dict[name] = adapters_modeling.Adapter(
                hidden_size=old_parent_module.dense.out_features,
                adapter_config=adapter_config,
            )
            layer_norm_dict[name] = modeling_bert.BertLayerNorm(
                normalized_shape=old_parent_module.LayerNorm.normalized_shape,
                eps=old_parent_module.LayerNorm.eps,
            )
        weighted_sum = WeightedSum(
            name_list=sub_module_name_list,
            mode=mode,
        )
        return cls(
            weighted_sum=weighted_sum,
            adapter_dict=adapter_dict,
            layer_norm_dict=layer_norm_dict,
            sub_module_name_list=sub_module_name_list,
        )


class FusedAdapters(nn.Module):
    def __init__(self,
                 module_name_list,
                 stacked_down_project,
                 activation,
                 up_project_weight, up_project_bias,
                 ):
        super().__init__()
        self.module_name_list = module_name_list
        # Linear: input_dim -> (k * adapter_dim)
        self.stacked_down_project = stacked_down_project
        self.activation = activation
        # weight dim: k, 1, adapter_dim, input_dim (yes this is weird)
        self.up_project_weight = nn.Parameter(up_project_weight)
        # bias dim: k, 1, 1, input_dim
        self.up_project_bias = nn.Parameter(up_project_bias)

        # Weirdly inverted
        self.input_dim = up_project_weight.shape[3]
        self.adapter_dim = up_project_weight.shape[2]
        self.k_dim = up_project_weight.shape[0]

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        # input: [batch_size, seq_len, input_dim]
        # output: [batch_size, seq_len, k, input_dim]
        down_projected = self.stacked_down_project(hidden_states)
        # => [batch_size, seq_len, adapter_dim]

        activated = self.activation(down_projected)
        # => [batch_size, seq_len, adapter_dim]

        activated = activated \
            .view(batch_size, seq_len, self.k_dim, self.adapter_dim)\
            .permute(2, 0, 1, 3)
        # => [k_dim, batch_size, seq_len, adapter_dim]

        up_projected = torch.matmul(activated, self.up_project_weight) + self.up_project_bias
        # => [k_dim, batch_size, seq_len, input_dim]

        final = (hidden_states.unsqueeze(0) + up_projected) \
            .permute(1, 2, 0, 3)
        # => [batch_size, seq_len, k_dim, input_dim]
        return final

    @classmethod
    def create(cls, adapter_dict, module_name_list=None):
        first_adapter = list(adapter_dict.values())[0]
        input_dim = first_adapter.hidden_size
        adapter_dim = first_adapter.adapter_config.adapter_size
        if module_name_list is None:
            module_name_list = list(adapter_dict.keys())
        k = len(adapter_dict)

        # Linear: input_dim -> (k * adapter_dim)
        stacked_down_project = nn.Linear(input_dim, k * adapter_dim)
        activation = first_adapter.activation
        # weight dim: k, 1, adapter_dim, input_dim (yes this is weird)
        up_project_weight = nn.Parameter(torch.randn(k, 1, adapter_dim, input_dim))
        # bias dim: k, 1, 1, input_dim
        up_project_bias = nn.Parameter(torch.randn(k, 1, 1, input_dim))
        return cls(
            module_name_list=module_name_list,
            stacked_down_project=stacked_down_project,
            activation=activation,
            up_project_weight=up_project_weight,
            up_project_bias=up_project_bias,
        )

    def load_weights(self, adapter_dict):
        a_dim = self.adapter_dim

        for i, name in enumerate(self.module_name_list):
            adapter = adapter_dict[name]
            self.stacked_down_project.weight.data[a_dim * i:a_dim * (i + 1), :] = \
                adapter.down_project.weight.data
            self.stacked_down_project.bias.data[a_dim * i:a_dim * (i + 1)] = \
                adapter.down_project.bias.data
            self.up_project_weight.data[i, 0] = adapter.up_project.weight.data.t()
            self.up_project_bias.data[i, 0, 0] = adapter.up_project.bias.data
        self.stacked_down_project.weight.data = self.stacked_down_project.weight.data.contiguous()
        self.up_project_weight.data = self.up_project_weight.data.contiguous()
        self.up_project_bias.data = self.up_project_bias.data.contiguous()

    @classmethod
    def create_and_load_weights(cls, adapter_dict, module_name_list=None):
        fused_adapter = cls.create(
            adapter_dict=adapter_dict,
            module_name_list=module_name_list,
        )
        fused_adapter.load_weights(adapter_dict)
        return fused_adapter


def create_fused_layer_norms(layer_norm_dict, module_name_list):
    dim_size = list(layer_norm_dict.values())[0].normalized_shape[0]
    num_modules = len(layer_norm_dict)
    fused_layer_norm = nn.LayerNorm((num_modules, dim_size))
    for i, k in enumerate(module_name_list):
        layer_norm = layer_norm_dict[k]
        fused_layer_norm.weight.data[i] = layer_norm.weight.data
        fused_layer_norm.bias.data[i] = layer_norm.bias.data
    fused_layer_norm.weight.data = fused_layer_norm.weight.data.contiguous()
    fused_layer_norm.bias.data = fused_layer_norm.bias.data.contiguous()
    return fused_layer_norm


class MultiAdapterOptimized(nn.Module):
    def __init__(self, weighted_sum, adapter_dict, layer_norm_dict,
                 sub_module_name_list=None):
        super(MultiAdapterOptimized, self).__init__()
        self.weighted_sum = weighted_sum
        self.adapter_dict = nn.ModuleDict(adapter_dict)
        self.layer_norm_dict = layer_norm_dict
        if sub_module_name_list is not None:
            self.sub_module_name_list = sub_module_name_list
        else:
            self.sub_module_name_list = list(adapter_dict.keys())

        dim_size = list(layer_norm_dict.values())[0].normalized_shape[0]
        num_modules = len(layer_norm_dict)
        self.fused_layer_norm = nn.LayerNorm((num_modules, dim_size))
        for i, (k, layer_norm) in enumerate(self.layer_norm_dict.items()):
            self.fused_layer_norm.weight.data[i] = layer_norm.weight.data
            self.fused_layer_norm.bias.data[i] = layer_norm.bias.data
        self.fused_layer_norm.weight.data = self.fused_layer_norm.weight.data.contiguous()
        self.fused_layer_norm.bias.data = self.fused_layer_norm.bias.data.contiguous()

    def forward(self, hidden_states, input_tensor):
        hidden_states_ls = []
        for k in self.module_name_ls:
            hidden_states_ls.append(self.adapter_dict[k](hidden_states))
        stacked_hidden_states = torch.stack(hidden_states_ls, dim=-2)
        broadcast_input_tensor = input_tensor.unsqueeze(-2)
        hidden_states = self.fused_layer_norm(broadcast_input_tensor + stacked_hidden_states)
        weights = self.weighted_sum.compute_weights().view(1, 1, -1, 1)
        hidden_states = (weights * hidden_states).sum(-2)
        return hidden_states

    @classmethod
    def create(cls,
               old_parent_module,
               sub_module_name_list,
               adapter_config: adapters_modeling.AdapterConfig,
               mode="softmax",
               include_base=True):
        adapter_dict = {}
        layer_norm_dict = {}
        if include_base:
            adapter_dict["base"] = torch_utils.IdentityModule()
            layer_norm_dict["base"] = old_parent_module.LayerNorm
        for name in sub_module_name_list:
            adapter_dict[name] = adapters_modeling.Adapter(
                hidden_size=old_parent_module.dense.out_features,
                adapter_config=adapter_config,
            )
            layer_norm_dict[name] = modeling_bert.BertLayerNorm(
                normalized_shape=old_parent_module.LayerNorm.normalized_shape,
                eps=old_parent_module.LayerNorm.eps,
            )
        weighted_sum = WeightedSum(
            name_list=list(adapter_dict.keys()),
            mode=mode,
        )
        return cls(
            weighted_sum=weighted_sum,
            adapter_dict=adapter_dict,
            layer_norm_dict=layer_norm_dict,
        )


class BertOutputWithMultiAdapters(nn.Module):
    def __init__(self, dense, multi_adapter, dropout):
        super(BertOutputWithMultiAdapters, self).__init__()
        self.dense = dense
        self.multi_adapter = multi_adapter
        self.dropout = dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.multi_adapter(
            hidden_states=hidden_states,
            input_tensor=input_tensor,
        )
        return hidden_states

    @classmethod
    def from_original(
        cls,
        old_module,
        sub_module_name_list,
        adapter_config: adapters_modeling.AdapterConfig,
        mode="softmax",
        include_base=True,
    ):
        assert isinstance(old_module, modeling_bert.BertOutput)
        multi_adapter = MultiAdapter.create(
            old_parent_module=old_module,
            sub_module_name_list=sub_module_name_list,
            adapter_config=adapter_config,
            mode=mode,
            include_base=include_base,
        )
        return cls(
            dense=old_module.dense,
            multi_adapter=multi_adapter,
            dropout=old_module.dropout,
        )


class BertSelfOutputWithMultiAdapters(nn.Module):
    def __init__(self, dense, multi_adapter, dropout):
        super(BertSelfOutputWithMultiAdapters, self).__init__()
        self.dense = dense
        self.multi_adapter = multi_adapter
        self.dropout = dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.multi_adapter(
            hidden_states=hidden_states,
            input_tensor=input_tensor,
        )
        return hidden_states

    @classmethod
    def from_original(
        cls,
        old_module,
        sub_module_name_list,
        adapter_config: adapters_modeling.AdapterConfig,
        mode="softmax",
        include_base=True,
    ):
        assert isinstance(old_module, modeling_bert.BertSelfOutput)
        multi_adapter = MultiAdapter.create(
            old_parent_module=old_module,
            sub_module_name_list=sub_module_name_list,
            adapter_config=adapter_config,
            mode=mode,
            include_base=include_base,
        )
        return cls(
            dense=old_module.dense,
            multi_adapter=multi_adapter,
            dropout=old_module.dropout,
        )


def add_multi_adapters(model, sub_module_name_list, adapter_config,
                       mode="softmax",
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
                    mode=mode,
                    include_base=include_base,
                )
                setattr(p_module, c_name, new_module)
                modified_layers[f"{p_name}.{c_name}"] = new_module.multi_adapter
            elif isinstance(c_module, modeling_bert.BertSelfOutput):
                p_name = p_name.split(".", 1)[1]
                new_module = BertSelfOutputWithMultiAdapters.from_original(
                    old_module=c_module,
                    sub_module_name_list=sub_module_name_list,
                    adapter_config=adapter_config,
                    mode=mode,
                    include_base=include_base,
                )
                setattr(p_module, c_name, new_module)
                modified_layers[f"{p_name}.{c_name}"] = new_module.multi_adapter
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
                """
                module_state_dict[f"layer_norm_dict.{adapter_set_name}.weight"] = \
                    weights_dict[f"{name}.LayerNorm.weight"]
                module_state_dict[f"layer_norm_dict.{adapter_set_name}.bias"] = \
                    weights_dict[f"{name}.LayerNorm.bias"]
                """
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
        weights = layer.weighted_sum.compute_weights().data.clone().cpu().numpy()
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
