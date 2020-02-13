import torch
import torch.nn as nn

import transformers.modeling_albert as modeling_albert


class AlbertForMultipleChoice(modeling_albert.AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.albert = modeling_albert.AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        num_choices = input_ids.shape[1]
        logits_list = []
        outputs_list = []
        for i in range(num_choices):
            outputs = self.albert(
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
        reshaped_logits = torch.cat([
            logits.unsqueeze(1).squeeze(-1)
            for logits in logits_list
        ], dim=1)
        reshaped_outputs = tuple([
            outputs[2:]
            for outputs in outputs_list
        ])

        outputs = (reshaped_logits,) + reshaped_outputs  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)
