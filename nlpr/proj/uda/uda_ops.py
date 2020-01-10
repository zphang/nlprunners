import numpy as np

import torch
import torch.nn.functional as F
from nlpr.shared.modeling.models import forward_batch_basic

import zproto.zlogv1 as zlogv1


def sup_train_step(model, sup_batch,
                   task, global_step, train_schedule, uda_params,
                   zlogger=zlogv1.VOID_LOGGER):
    # Compute logits, logprobs
    sup_logits = forward_batch_basic(model=model, batch=sup_batch, omit_label_id=True)[0]

    sup_loss = compute_sup_loss(
        sup_logits=sup_logits,
        label_id=sup_batch.label_id,
        task=task, global_step=global_step,
        train_schedule=train_schedule, uda_params=uda_params,
        zlogger=zlogger,
    )
    return sup_loss, sup_logits


def compute_sup_loss(sup_logits, label_id,
                     task, global_step, train_schedule, uda_params,
                     zlogger=zlogv1.VOID_LOGGER):
    # Compute cross entropy (why manually? to get the mask I guess)
    per_example_loss = F.cross_entropy(sup_logits, label_id, reduction="none")
    # Create mask-template
    loss_mask = torch.ones(
        per_example_loss.size(),
        dtype=per_example_loss.dtype,
        device=per_example_loss.device,
    )

    # Using TSA (Training Signal Annealing)
    if uda_params.tsa:
        # Compute probability of predicting correct label
        sup_logprobs = F.log_softmax(sup_logits, dim=-1)
        one_hot_labels = F.one_hot(label_id, num_classes=len(task.LABELS)).float()
        correct_label_probs = (one_hot_labels * torch.exp(sup_logprobs)).sum(dim=-1)
        # TSA weight lower-bounded by 1/K
        tsa_threshold = get_tsa_threshold(
            schedule=uda_params.tsa_schedule,
            global_step=global_step,
            num_train_steps=train_schedule.t_total,
            start=1 / len(task.LABELS),
            end=1.,
        )
        # Only backprop loss if predicted probability is LOWER than tsa threshold
        larger_than_threshold = correct_label_probs > tsa_threshold
        loss_mask = loss_mask * (1 - larger_than_threshold.float())

        zlogger.write_entry("uda_metrics", {"tsa_threshold": float(tsa_threshold)})
        zlogger.write_entry("uda_metrics", {"sup_loss_mask": float(loss_mask.mean().item())})

    # IMPORTANT: Don't backprop through the mask
    loss_mask = loss_mask.detach()
    per_example_loss = per_example_loss * loss_mask
    # Supervised loss only on TSA-masked elements
    # (Correct the denominator, also prevent division by 0)
    sup_loss = per_example_loss.sum() / loss_mask.sum().clamp(min=1)
    return sup_loss


def unsup_train_step(model, unsup_orig_batch, unsup_aug_batch, uda_params,
                     zlogger=zlogv1.VOID_LOGGER):
    # Compute Logits
    unsup_orig_logits = forward_batch_basic(model=model, batch=unsup_orig_batch, omit_label_id=True)[0]
    unsup_aug_logits = forward_batch_basic(model=model, batch=unsup_aug_batch, omit_label_id=True)[0]

    unsup_loss = compute_unsup_loss(
        unsup_orig_logits=unsup_orig_logits,
        unsup_aug_logits=unsup_aug_logits,
        uda_params=uda_params,
        zlogger=zlogger,
    )
    return unsup_loss, unsup_orig_logits, unsup_aug_logits


def compute_unsup_loss(unsup_orig_logits, unsup_aug_logits, uda_params,
                       zlogger=zlogv1.VOID_LOGGER):
    # Compute logprobs
    #   Use regular softmax (-1) or use temperature for softmax (not used for NLP)
    if uda_params.uda_softmax_temp != -1:
        unsup_orig_logprobs = F.log_softmax(
            unsup_orig_logits / uda_params.uda_softmax_temp,
            dim=-1,
        ).detach()
    else:
        unsup_orig_logprobs = F.log_softmax(unsup_orig_logits, dim=-1).detach()
    unsup_aug_logprobs = F.log_softmax(unsup_aug_logits, dim=-1)

    # Compute threshold mask
    #   UDA loss only if largest predicted probability > threshold (not used for NLP)
    if uda_params.uda_confidence_thresh != -1:
        largest_prob = unsup_orig_logprobs.max(dim=-1)[0]
        unsup_loss_mask = (largest_prob > uda_params.uda_confidence_thresh).float().detach()
        zlogger.write_entry("uda_metrics", {"unsup_loss_mask": float(unsup_loss_mask.mean().item())})
    else:
        unsup_loss_mask = 1

    # Compute KL between orig and augmented
    per_example_kl_loss = (
        kl_for_log_probs(unsup_orig_logprobs, unsup_aug_logprobs)
        * unsup_loss_mask
    )
    # Compute loss
    unsup_loss = per_example_kl_loss.mean()

    return unsup_loss


def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
    training_progress = float(global_step) / float(num_train_steps)
    assert 0 <= training_progress <= 1
    if schedule == "no_schedule":
        threshold = 1.
    elif schedule == "linear_schedule":
        threshold = training_progress
    elif schedule == "exp_schedule":
        scale = 5
        threshold = np.exp((training_progress - 1) * scale)
        # [exp(-5), exp(0)] = [1e-2, 1]
    elif schedule == "log_schedule":
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        threshold = 1 - np.exp((-training_progress) * scale)
    else:
        raise KeyError(schedule)
    return threshold * (end - start) + start


def kl_for_log_probs(log_p, log_q):
    p = torch.exp(log_p)
    neg_ent = (p * log_p).sum(dim=-1)
    neg_cross_ent = (p * log_q).sum(dim=-1)
    kl = neg_ent - neg_cross_ent
    return kl
