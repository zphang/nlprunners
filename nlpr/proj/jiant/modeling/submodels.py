import abc
from dataclasses import dataclass

import torch
import torch.nn as nn
import nlpr.proj.jiant.modeling.heads as heads


class Submodel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, batch, task, tokenizer, compute_loss: bool = False):
        raise NotImplementedError


class ClassificationModel(Submodel):
    def __init__(self, encoder, classification_head: heads.ClassificationHead):
        super().__init__(encoder=encoder)
        self.classification_head = classification_head

    def forward(self, batch, task, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(
            encoder=self.encoder,
            batch=batch,
        )
        logits = self.classification_head(pooled=encoder_output.pooled)
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.classification_head.num_labels),
                batch.label_id.view(-1),
            )
            return logits, loss
        else:
            return logits


class RegressionModel(Submodel):
    def __init__(self, encoder, regression_head: heads.RegressionHead):
        super().__init__(encoder=encoder)
        self.regression_head = regression_head

    def forward(self, batch, task, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(
            encoder=self.encoder,
            batch=batch,
        )
        scores = self.regression_head(pooled=encoder_output.pooled)
        if compute_loss:
            loss_fct = nn.MSELoss()
            loss = loss_fct(scores.view(-1), batch.label_id.view(-1))
            return scores, loss
        else:
            return scores


class MultipleChoiceModel(Submodel):
    def __init__(self, encoder, num_choices: int, choice_scoring_head: heads.RegressionHead):
        super().__init__(encoder=encoder)
        self.num_choices = num_choices
        self.choice_scoring_head = choice_scoring_head

    def forward(self, batch, task, tokenizer, compute_loss: bool = False):
        input_ids = batch.input_ids
        segment_ids = batch.segment_ids
        input_mask = batch.input_mask

        choice_score_list = []
        for i in range(self.num_choices):
            encoder_output = get_output_from_encoder(
                encoder=self.encoder,
                input_ids=input_ids[:, i],
                segment_ids=segment_ids[:, i],
                input_mask=input_mask[:, i],
            )
            choice_score = self.choice_scoring_head(pooled=encoder_output.pooled)
            choice_score_list.append(choice_score)

        logits = torch.cat([
            choice_score.unsqueeze(1).squeeze(-1)
            for choice_score in choice_score_list
        ], dim=1)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.choice_scoring_head.num_labels),
                batch.label_ids.view(-1),
            )
            return logits, loss
        else:
            return logits


class SpanComparisonModel(Submodel):
    def __init__(self, encoder,
                 span_comparison_head: heads.SpanComparisonHead):
        super().__init__(encoder=encoder)
        self.span_comparison_head = span_comparison_head

    def forward(self, batch, task, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(
            encoder=self.encoder,
            batch=batch,
        )
        logits = self.span_comparison_head(
            unpooled=encoder_output.unpooled,
            spans=batch.spans,
        )
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.span_comparison_head.num_labels),
                batch.label_id.view(-1),
            )
            return logits, loss
        else:
            return logits


class TokenClassificationModel(Submodel):
    def __init__(self, encoder,
                 token_classification_head: heads.TokenClassificationHead):
        super().__init__(encoder=encoder)
        self.token_classification_head = token_classification_head

    def forward(self, batch, task, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(
            encoder=self.encoder,
            batch=batch,
        )
        logits = self.span_comparison_head(
            unpooled=encoder_output.unpooled,
            spans=batch.spans,
        )
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.token_classification_head.num_labels),
                batch.label_ids.view(-1),
            )
            return logits, loss
        else:
            return logits


class QAModel(Submodel):
    def __init__(self, encoder, qa_head: heads.QAHead):
        super().__init__(encoder=encoder)
        self.qa_head = qa_head

    def forward(self, batch, task, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(
            encoder=self.encoder,
            batch=batch,
        )
        logits = self.qa_head(unpooled=encoder_output.unpooled)
        if compute_loss:
            loss = compute_qa_loss(
                logits=logits,
                start_positions=batch.start_position,
                end_positions=batch.end_position,
            )
            return logits, loss
        else:
            return logits


class MLMModel(Submodel):
    def __init__(self, encoder, mlm_head: heads.BaseMLMHead):
        super().__init__(encoder=encoder)
        self.mlm_head = mlm_head

    def forward(self, batch, task, tokenizer, compute_loss: bool = False):
        masked_batch = batch.get_masked(
            mlm_probability=task.mlm_probability,
            tokenizer=tokenizer,
        )
        encoder_output = get_output_from_encoder_and_batch(
            encoder=self.encoder,
            batch=masked_batch,
        )
        logits = self.mlm_head(unpooled=encoder_output.unpooled)
        if compute_loss:
            loss = compute_mlm_loss(
                logits=logits,
                masked_lm_labels=masked_batch.masked_lm_labels,
            )
            return logits, loss
        else:
            return logits


@dataclass
class EncoderOutput:
    pooled: torch.Tensor
    unpooled: torch.Tensor
    # Extend later with attention, hidden_acts, etc


def get_output_from_encoder_and_batch(encoder, batch) -> EncoderOutput:
    return get_output_from_encoder(
        encoder=encoder,
        input_ids=batch.input_ids,
        segment_ids=batch.segment_ids,
        input_mask=batch.input_mask,
    )


def get_output_from_encoder(encoder, input_ids, segment_ids, input_mask) -> EncoderOutput:
    output = encoder(
        input_ids=input_ids,
        token_type_ids=segment_ids,
        attention_mask=input_mask,
    )
    # Extend later with attention, hidden_acts, etc
    assert len(output) == 2
    return EncoderOutput(
        pooled=output[1],
        unpooled=output[0],
    )


def compute_mlm_loss(logits, masked_lm_labels):
    vocab_size = logits.shape(-1)
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(
        logits.view(-1, vocab_size),
        masked_lm_labels.view(-1)
    )


def compute_qa_loss(logits, start_positions, end_positions):
    # Do we want to keep them as 1 tensor, or multiple?
    # bs x 2 x seq_len x 1

    start_logits, end_logits = logits[:, 0], logits[:, 1]
    # Taken from: RobertaForQuestionAnswering
    # If we are on multi-GPU, split add a dimension
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2
    return total_loss
