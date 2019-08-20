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


def compute_pred_entropy(logits):
    # logits are pre softmax
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    return -(p * log_p).sum(dim=-1).mean()


def compute_pred_entropy_clean(logits):
    return float(compute_pred_entropy(logits).item())
