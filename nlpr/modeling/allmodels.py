import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor

from pytorch_transformers import (
    BertPreTrainedModel, BertModel, BertForSequenceClassification
)

import nlpr.tasks as tasks


class BertForJointMultipleChoice(BertPreTrainedModel):
    def __init__(self, config, num_choices):
        """
        I think the actual BertForMultipleChoice assumes a specific kind of data batching
         which is unclear to the user
        """
        super(BertForJointMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # BertForMultipleChoice version
        """
        self.classifier = nn.Linear(config.hidden_size, 1)
        """

        # New Version
        self.classifier = nn.Linear(config.hidden_size * num_choices, num_choices)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids_list, token_type_ids_list=None, attention_mask_list=None,
                labels=None):
        assert len(input_ids_list) == self.num_choices
        if token_type_ids_list is None:
            token_type_ids_list = [None] * self.num_choices
        if attention_mask_list is None:
            attention_mask_list = [None] * self.num_choices

        bert_output_ls = [
            self.bert(input_ids, token_type_ids, attention_mask,
                      output_all_encoded_layers=False, return_attn=False)
            for input_ids, token_type_ids, attention_mask
            in zip(input_ids_list, token_type_ids_list, attention_mask_list)
        ]

        # BertForMultipleChoice version
        """
        logits_ls = [
            self.classifier(self.dropout(pooled_output))
            for _, pooled_output
            in bert_output_ls
        ]
        logits = torch.cat(logits_ls, dim=-1)
        """

        # New Version
        pooled_output_ls = [
            self.dropout(pooled_output)
            for _, pooled_output
            in bert_output_ls
        ]
        cat_pooled_output = torch.cat(pooled_output_ls, dim=-1)
        logits = self.classifier(cat_pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_choices), labels.view(-1))
            return loss
        else:
            return logits


class BertForSpanComparisonClassification(BertPreTrainedModel):
    def __init__(self, config, num_spans, num_labels):
        super(BertForSpanComparisonClassification, self).__init__(config)
        self.num_spans = num_spans
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.span_attention_extractor = SelfAttentiveSpanExtractor(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size * self.num_spans, self.num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, spans,
                token_type_ids=None, attention_mask=None,
                labels=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask,
            output_all_encoded_layers=False,
        )
        span_embeddings = self.span_attention_extractor(sequence_output, spans)
        span_embeddings = span_embeddings.view(-1, self.num_spans * self.config.hidden_size)
        span_embeddings = self.dropout(span_embeddings)

        logits = self.classifier(span_embeddings)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class TaskModel:
    def forward_batch(self, batch):
        raise NotImplementedError

    def forward_batch_hide_label(self, batch):
        raise NotImplementedError


class CBModel(BertForSequenceClassification, TaskModel):
    def __init__(self, config):
        super().__init__(config, num_labels=3)

    def forward_batch(self, batch):
        return self(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_ids,
        )

    def forward_batch_hide_label(self, batch):
        return self.forward_batch(batch.new(label_ids=None))


class CopaModel(BertForJointMultipleChoice, TaskModel):
    def __init__(self, config):
        super().__init__(config, num_choices=2)

    def forward_batch(self, batch):
        return self(
            input_ids_list=[batch.input_ids1, batch.input_ids2],
            token_type_ids_list=[batch.segment_ids1, batch.segment_ids2],
            attention_mask_list=[batch.input_mask1, batch.input_mask2],
            labels=batch.label_ids,
        )

    def forward_batch_hide_label(self, batch):
        return self.forward_batch(batch.new(label_ids=None))


class MultiRCModel(BertForSequenceClassification, TaskModel):
    def __init__(self, config):
        super().__init__(config, num_labels=2)

    def forward_batch(self, batch):
        return self(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_ids,
        )

    def forward_batch_hide_label(self, batch):
        return self.forward_batch(batch.new(label_ids=None))


class RTEModel(BertForSequenceClassification, TaskModel):
    def __init__(self, config):
        super().__init__(config, num_labels=2)

    def forward_batch(self, batch):
        return self(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_ids,
        )

    def forward_batch_hide_label(self, batch):
        return self.forward_batch(batch.new(label_ids=None))


class WSCModel(BertForSpanComparisonClassification, TaskModel):
    def __init__(self, config):
        super().__init__(config, num_spans=2, num_labels=2)

    def forward_batch(self, batch):
        spans = torch.stack([
            batch.span1_span,
            batch.span2_span,
        ], dim=-2)
        return self(
            input_ids=batch.input_ids,
            spans=spans,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_ids,
        )

    def forward_batch_hide_label(self, batch):
        return self.forward_batch(batch.new(label_ids=None))


class WiCModel(BertForSpanComparisonClassification, TaskModel):
    def __init__(self, config):
        super().__init__(config, num_spans=2, num_labels=2)

    def forward_batch(self, batch):
        spans = torch.stack([
            batch.sent1_span,
            batch.sent2_span,
        ], dim=-2)
        return self(
            input_ids=batch.input_ids,
            spans=spans,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_ids,
        )

    def forward_batch_hide_label(self, batch):
        return self.forward_batch(batch.new(label_ids=None))


class YelpPolarityModel(BertForSequenceClassification, TaskModel):
    def __init__(self, config):
        super().__init__(config, num_labels=2)

    def forward_batch(self, batch):
        return self(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_ids,
        )

    def forward_batch_hide_label(self, batch):
        return self.forward_batch(batch.new(label_ids=None))


class AmazonPolarityModel(BertForSequenceClassification, TaskModel):
    def __init__(self, config):
        super().__init__(config, num_labels=2)

    def forward_batch(self, batch):
        return self(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_ids,
        )

    def forward_batch_hide_label(self, batch):
        return self.forward_batch(batch.new(label_ids=None))


class MNLIModel(BertForSequenceClassification, TaskModel):
    def __init__(self, config):
        super().__init__(config, num_labels=3)

    def forward_batch(self, batch):
        return self(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_ids,
        )

    def forward_batch_hide_label(self, batch):
        return self.forward_batch(batch.new(label_ids=None))


class IMDBModel(BertForSequenceClassification, TaskModel):
    def __init__(self, config):
        super().__init__(config, num_labels=2)

    def forward_batch(self, batch):
        return self(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=batch.label_ids,
        )

    def forward_batch_hide_label(self, batch):
        return self.forward_batch(batch.new(label_ids=None))


def map_task_to_model_class(task):
    if isinstance(task, tasks.CommitmentBankTask):
        model_class = CBModel
    elif isinstance(task, tasks.CopaTask):
        model_class = CopaModel
    elif isinstance(task, tasks.MultiRCTask):
        model_class = MultiRCModel
    elif isinstance(task, tasks.RteTask):
        model_class = RTEModel
    elif isinstance(task, tasks.WSCTask):
        model_class = WSCModel
    elif isinstance(task, tasks.WiCTask):
        model_class = WiCModel
    elif isinstance(task, tasks.YelpPolarityTask):
        model_class = YelpPolarityModel
    elif isinstance(task, tasks.AmazonPolarityTask):
        model_class = AmazonPolarityModel
    elif isinstance(task, tasks.MnliTask):
        model_class = MNLIModel
    elif isinstance(task, tasks.IMDBTask):
        model_class = IMDBModel
    else:
        raise KeyError(task)
    return model_class
