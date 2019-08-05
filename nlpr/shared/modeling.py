import torch.nn as nn


def forward_batch_basic(model: nn.Module, batch, omit_label_ids: bool = False):
    return model(
        input_ids=batch.input_ids,
        token_type_ids=batch.segment_ids,
        attention_mask=batch.input_mask,
        labels=batch.label_ids if not omit_label_ids else None,
    )
