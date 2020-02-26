import torch
import torch.nn as nn

import transformers as ptt
import transformers.modeling_roberta as modeling_roberta


class RobertaForSpanChoiceProb(ptt.BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForSpanChoiceProb, self).__init__(config)

        self.roberta = ptt.RobertaModel(config)
        self.lm_head = modeling_roberta.RobertaLMHead(config)

        self.init_weights()
        self.tie_weights()

    def forward(self,
                input_ids_1, token_type_ids_1, attention_mask_1,
                input_ids_2, token_type_ids_2, attention_mask_2,
                span_1, span_2,
                labels=None):
        prediction_scores_1 = self.get_prediction_scores(
            input_ids=input_ids_1,
            token_type_ids=token_type_ids_1,
            attention_mask=attention_mask_1,
        )
        raise NotImplementedError("lol todo")

    def get_prediction_scores(self, input_ids, token_type_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            head_mask=None,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        return prediction_scores


class RobertaForMultipleChoice(ptt.BertPreTrainedModel):
    config_class = ptt.RobertaConfig
    pretrained_model_archive_map = ptt.ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = ptt.RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        num_choices = input_ids.shape[1]

        logits_list = []
        misc_list = []
        for i in range(num_choices):
            outputs = self.roberta(
                input_ids[:, i],
                position_ids=position_ids[:, i] if position_ids is not None else None,
                token_type_ids=token_type_ids[:, i] if token_type_ids is not None else None,
                attention_mask=attention_mask[:, i] if attention_mask is not None else None,
                head_mask=head_mask,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            logits_list.append(logits)
            misc_list.append(outputs[2:])

        reshaped_logits = torch.cat([
            logits.unsqueeze(1).squeeze(-1)
            for logits in logits_list
        ], dim=1)

        reshaped_outputs = []
        for j in range(len(misc_list[0])):
            reshaped_outputs.append([
                torch.stack([misc[j][layer_i] for misc in misc_list], dim=1)
                for layer_i in range(len(misc_list[0][0]))
            ])
        reshaped_outputs = tuple(reshaped_outputs)

        outputs = (reshaped_logits,) + reshaped_outputs  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)
