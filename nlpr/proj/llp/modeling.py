from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlpr.shared.model_resolution import ModelArchitectures


def get_ptt_model_embedding_dim(ptt_model):
    model_arch = ModelArchitectures.from_ptt_model(ptt_model)
    if model_arch in (ModelArchitectures.BERT,
                      ModelArchitectures.XLNET,
                      ModelArchitectures.XLM,
                      ModelArchitectures.ROBERTA):
        return ptt_model.config.hidden_size
    elif model_arch == ModelArchitectures.GLOVE_LSTM:
        return ptt_model.model.hidden_dim
    else:
        raise KeyError(model_arch)


@dataclass
class _InputSet:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    input_mask: torch.Tensor


@dataclass
class _Output:
    logits: torch.Tensor
    embedding: torch.Tensor


class LlpModel(nn.Module):
    def __init__(self, ptt_model, embedding_dim, dropout_p=0.5):
        super().__init__()
        self.ptt_model = ptt_model
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p

        self.model_arch = ModelArchitectures.from_ptt_model(ptt_model)
        self.embedding_layer = nn.Linear(
            get_ptt_model_embedding_dim(ptt_model),
            embedding_dim,
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward_batch(self, batch, normalize_embedding=True):
        return self(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            input_mask=batch.input_mask,
            normalize_embedding=normalize_embedding,
        )

    def forward(self, input_ids, token_type_ids, input_mask, normalize_embedding=True):
        # Todo: Assume classification for now, refactor later
        pooled_output, logits = self.get_pooled(_InputSet(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
        ))

        # pooled_output = self.dropout(pooled_output)

        #embedding = self.embedding_layer(F.relu(pooled_output))
        embedding = self.embedding_layer(pooled_output)
        if normalize_embedding:
            returned_embedding = F.normalize(embedding, p=2, dim=1)
        else:
            returned_embedding = embedding

        return _Output(logits=logits, embedding=returned_embedding)

    def get_pooled(self, input_set: _InputSet):
        # Will probably need to refactor this out later
        if self.model_arch == ModelArchitectures.BERT:
            return self._get_bert_pooled(input_set)
        elif self.model_arch == ModelArchitectures.XLNET:
            return self._get_xlnet_pooled(input_set)
        elif self.model_arch == ModelArchitectures.ROBERTA:
            return self._get_roberta_pooled(input_set)
        elif self.model_arch == ModelArchitectures.GLOVE_LSTM:
            return self._get_glove_lstm_pooled(input_set)
        else:
            raise KeyError(self.model_arch)

    def _get_bert_pooled(self, input_set: _InputSet):
        bert_output = self.ptt_model.bert(
            input_ids=input_set.input_ids,
            token_type_ids=input_set.token_type_ids,
            attention_mask=input_set.input_mask,
        )
        _, pooled_output = bert_output
        pooled_output = self.ptt_model.dropout(pooled_output)
        logits = self.ptt_model.classifier(pooled_output)
        return pooled_output, logits

    def _get_xlnet_pooled(self, input_set: _InputSet):
        transformer_outputs = self.transformer(
            input_ids=input_set.input_ids,
            token_type_ids=input_set.token_type_ids,
            input_mask=input_set.input_mask,
            # ^ yes, this makes no sense. I think it's a legacy naming issue
        )
        output = transformer_outputs[0]
        output = self.sequence_summary(output)
        logits = self.logits_proj(output)
        return output, logits

    def _get_xlm_pooled(self, input_set: _InputSet):
        transformer_outputs = self.transformer(
            input_ids=input_set.input_ids,
            token_type_ids=input_set.token_type_ids,
            input_mask=input_set.input_mask,
            # ^ yes, this makes no sense. I think it's a legacy naming issue
        )
        output = transformer_outputs[0]
        output = self.sequence_summary(output)
        logits = self.logits_proj(output)
        return output, logits

    def _get_roberta_pooled(self, input_set: _InputSet):
        roberta_output = self.ptt_model.roberta(
            input_ids=input_set.input_ids,
            token_type_ids=input_set.token_type_ids,
            attention_mask=input_set.input_mask,
        )
        sequence_output = roberta_output[0]
        logits = self.ptt_model.classifier(sequence_output)
        return sequence_output[:, 0], logits

    def _get_glove_lstm_pooled(self, input_set: _InputSet):
        logits, pooled = self.ptt_model(
            input_ids=input_set.input_ids,
            token_type_ids=input_set.token_type_ids,
            attention_mask=input_set.input_mask,
            labels=None,
        )
        # Notice the swapped order
        return pooled, logits
