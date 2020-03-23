import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

CPU_DEVICE = torch.device("cpu")


def normalize_embedding_tensor(embedding):
    return F.normalize(embedding, p=2, dim=1)


def embedding_norm_loss(raw_embedding):
    norms = raw_embedding.norm(dim=1)
    return F.mse_loss(norms, torch.ones_like(norms), reduction='none')


def get_val(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    else:
        return x


def compute_pred_entropy(logits):
    # logits are pre softmax
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    return -(p * log_p).sum(dim=-1).mean()


def compute_pred_entropy_clean(logits):
    return float(compute_pred_entropy(logits).item())


def copy_state_dict(state_dict, target_device=None):
    if target_device is None:
        return copy.deepcopy(state_dict)

    return {
        k: v.to(target_device).clone()
        for k, v in state_dict.items()
    }


def get_parent_child_module_list(model):
    ls = []
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            ls.append((parent_name, parent_module, child_name, child_module))
    return ls


class IdentityModule(nn.Module):
    def forward(self, *inputs):
        if len(inputs) == 1:
            return inputs[0]
        else:
            return inputs


def set_requires_grad(named_parameters, requires_grad):
    for name, param in named_parameters:
        param.requires_grad = requires_grad


def get_only_requires_grad(parameters, requires_grad=True):
    if isinstance(parameters, list):
        if not parameters:
            return []
        elif isinstance(parameters[0], tuple):
            return [(n, p) for n, p in parameters if p.requires_grad == requires_grad]
        else:
            return [p for p in parameters if p.requires_grad == requires_grad]
    elif isinstance(parameters, dict):
        return {n: p for n, p in parameters if p.requires_grad == requires_grad}
    else:
        raise RuntimeError("todo: support generators")


class ListDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class DataLoaderWithLength(DataLoader):
    def __len__(self):
        try:
            return super().__len__()
        except TypeError as e:
            try:
                return math.ceil(len(self.dataset) / self.batch_size)
            except TypeError:
                pass
            raise e


def safe_save(obj, path, temp_path=None):
    if temp_path is None:
        temp_path = path + "._temp"
    torch.save(obj, temp_path)
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)
