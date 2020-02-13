import torch.nn as nn
import transformers as ptt
from torch.nn import CrossEntropyLoss, MSELoss

from nlpr.ext.allennlp import SelfAttentiveSpanExtractor


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
