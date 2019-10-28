import torch

CPU = torch.device("cpu")


def extract_jiant_adapter_weights(loaded):
    jiant_adapter_weights = {
        k.replace("sent_encoder._text_field_embedder.model.", ""): v
        for k, v in loaded.items()
        if "sent_encoder._text_field_embedder.model." in k
    }
    return jiant_adapter_weights


def extract_jiant_adapter_weights_path(path):
    loaded = torch.load(path, map_location=CPU)
    return extract_jiant_adapter_weights(loaded)


def extract_nlpr_adapter_weights(loaded):
    nlpr_adapter_weights = {
        k.replace("roberta.", ""): v
        for k, v in loaded.items()
        if "roberta." in k and ("LayerNorm" in k or "adapter" in k or "pooler" in k) and "embeddings" not in k
    }

    return nlpr_adapter_weights


def extract_nlpr_adapter_weights_path(path):
    loaded = torch.load(path, map_location=CPU)
    return extract_nlpr_adapter_weights(loaded)
