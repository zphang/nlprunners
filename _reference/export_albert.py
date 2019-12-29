import transformers
import transformers.configuration_albert
import pyutils.io as io
import os
import torch
import tqdm


meta_config_base_path = "/home/zp489/scratch/persistent/191027_model_configs_cs"
cache_base_path = "/home/zp489/scratch/working/v1/1910/27_ptt_models/all_cache"
config_base_path = os.path.join(cache_base_path, "configs")
models_base_path = os.path.join(cache_base_path, "models")
tokenizers_base_path = os.path.join(cache_base_path, "tokenizers")


for albert_name in tqdm.tqdm_notebook(transformers.configuration_albert.ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP):
    os.makedirs(os.path.join(tokenizers_base_path, albert_name), exist_ok=True)
    tokenizer_path = os.path.join(tokenizers_base_path, albert_name)
    model_path = os.path.join(config_base_path, f"{albert_name}.p")
    model_config_path = os.path.join(config_base_path, f"{albert_name}.json")
    model = transformers.AlbertForMaskedLM.from_pretrained(albert_name)
    torch.save(model.state_dict(), model_path)
    io.write_json(model.config.to_dict(), model_config_path)
    tokenizer = transformers.AlbertTokenizer.from_pretrained(albert_name)
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    config = {
        "model_type": albert_name,
        "model_path": model_path,
        "model_config_path": model_config_path,
        "model_tokenizer_path": tokenizer_path,
    }
    io.write_json(config, os.path.join(meta_config_base_path, f"{albert_name}.json"))
