import torch.nn as nn

import transformers.modeling_roberta as modeling_roberta
from transformers import PreTrainedModel
from transformers.modeling_bert import BertLayerNorm


class RobertaPretrainedModel(PreTrainedModel):
    config_class = modeling_roberta.RobertaConfig
    pretrained_model_archive_map = modeling_roberta.ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, *inputs, **kwargs):
        super(RobertaPretrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
