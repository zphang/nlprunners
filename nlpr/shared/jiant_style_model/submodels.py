import abc

import torch.nn as nn
import nlpr.shared.jiant_style_model.heads as heads


class Submodel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, batch, task, tokenizer, compute_loss: bool = False):
        raise NotImplementedError


class ClassificationModel(Submodel):
    def __init__(self, encoder, classification_head):
        super().__init__(encoder=encoder)
        self.classification_head = classification_head

    def forward(self, batch, task, tokenizer, compute_loss: bool = False):
        pooled, unpooled

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), batch.label_ids.view(-1))
            return logits, loss
        else:
            return logits
