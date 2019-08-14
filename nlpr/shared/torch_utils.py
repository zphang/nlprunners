import torch
import torch.nn.functional as F


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
