import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return inputs
