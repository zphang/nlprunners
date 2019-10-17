from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

import transformers as ptt
from torch.nn import CrossEntropyLoss, MSELoss
from nlpr.tasks.lib.shared import TaskTypes
from nlpr.ext.allennlp import SelfAttentiveSpanExtractor


@dataclass
class InputSet:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor


def forward_batch_basic(model: nn.Module, batch, omit_label_ids: bool = False):
    return model(
        input_ids=batch.input_ids,
        token_type_ids=batch.segment_ids,
        attention_mask=batch.input_mask,
        labels=batch.label_ids if not omit_label_ids else None,
    )


def forward_batch_delegate(model: nn.Module, batch, task_type: TaskTypes, omit_label_ids: bool = False):
    if task_type in [TaskTypes.CLASSIFICATION, TaskTypes.REGRESSION]:
        return forward_batch_basic(
            model=model,
            batch=batch,
            omit_label_ids=omit_label_ids,
        )
    elif task_type == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        spans = torch.stack([
            batch.sentence1_span,
            batch.sentence2_span,
        ], dim=-2)
        return model(
            input_ids=batch.input_ids,
            spans=spans,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_ids if not omit_label_ids else None,
        )
    elif task_type == TaskTypes.MULTIPLE_CHOICE:
        assert hasattr(model, "num_choices")
        input_set_list = []
        # Todo: huge hack. Expose as method?
        for i in range(1, model.num_choices + 1):
            input_set_list.append(InputSet(
                input_ids=getattr(batch, f"input_ids{i}"),
                token_type_ids=getattr(batch, f"input_mask{i}"),
                attention_mask=getattr(batch, f"segment_ids{i}"),
            ))
        return model(
            input_set_list=input_set_list,
            labels=batch.label_ids if not omit_label_ids else None,
        )
    else:
        raise KeyError(task_type)


def compute_loss_from_model_output(logits, loss_criterion, batch, task_type: TaskTypes):
    # todo: cleanup
    if task_type == TaskTypes.CLASSIFICATION:
        loss = loss_criterion(logits, batch.label_ids)
    elif task_type == TaskTypes.REGRESSION:
        loss = loss_criterion(logits.squeeze(-1), batch.label)
    elif task_type == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        loss = loss_criterion(logits, batch.label_ids)
    else:
        raise KeyError(task_type)
    return loss


class BertForSequenceRegression(ptt.BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSequenceRegression, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = ptt.BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForSpanComparisonClassification(ptt.BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSpanComparisonClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_spans = config.num_spans

        self.bert = ptt.BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.span_attention_extractor = SelfAttentiveSpanExtractor(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size * self.num_spans, self.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, spans,
                token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        bert_output = self.bert(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            attention_mask=attention_mask, head_mask=head_mask,
        )
        sequence_output = bert_output[0]
        span_embeddings = self.span_attention_extractor(sequence_output, spans)
        span_embeddings = span_embeddings.view(-1, self.num_spans * self.config.hidden_size)
        span_embeddings = self.dropout(span_embeddings)

        logits = self.classifier(span_embeddings)

        outputs = (logits,) + bert_output[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForMultipleChoice(ptt.BertPreTrainedModel):

    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = config.num_choices

        self.bert = ptt.BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.scorer = nn.Linear(config.hidden_size, 1)

        self.apply(self.init_weights)

    def forward(self,
                input_set_list: List[InputSet],
                labels=None,
                ):
        # Uses num_choices BERTs worth of computation
        logits_list = []
        for input_set in input_set_list:
            outputs = self.bert(
                input_ids=input_set.input_ids,
                token_type_ids=input_set.token_type_ids,
                attention_mask=input_set.attention_mask,
                position_ids=None,
                head_mask=None,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.scorer(pooled_output).squeeze(1)
            logits_list.append(logits)

        combined_logits = torch.cat(logits_list, dim=1)

        return combined_logits, None, None


class RoBertaForMultipleChoice(ptt.BertPreTrainedModel):

    def __init__(self, config):
        super(RoBertaForMultipleChoice, self).__init__(config)
        self.num_choices = config.num_choices

        self.roberta = ptt.RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.scorer = nn.Linear(config.hidden_size, 1)

    def forward(self,
                input_set_list: List[InputSet],
                labels=None,
                ):
        # Uses num_choices BERTs worth of computation
        logits_list = []
        for input_set in input_set_list:
            outputs = self.roberta(
                input_ids=input_set.input_ids,
                token_type_ids=input_set.token_type_ids,
                attention_mask=input_set.attention_mask,
                position_ids=None,
                head_mask=None,
            )
            sequence_output = outputs[0]
            logits = self.scorer(sequence_output).squeeze(1)
            logits_list.append(logits)

        combined_logits = torch.cat(logits_list, dim=1)

        return combined_logits, None, None
