import os

import torch
import transformers as tfm

import pyutils.io as io
import zconf


CLASS_LOOKUP = {
    "bert": (tfm.BertForSequenceClassification, tfm.BertTokenizer),
    "xlnet": (tfm.XLNetForSequenceClassification, tfm.XLNetTokenizer),
    "xlm": (tfm.XLMForSequenceClassification, tfm.XLMTokenizer),
    "roberta": (tfm.RobertaForSequenceClassification, tfm.RobertaTokenizer),
    "distilbert": (tfm.DistilBertForSequenceClassification, tfm.DistilBertTokenizer),
    "albert": (tfm.AlbertForSequenceClassification, tfm.AlbertTokenizer),
}


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    model_type = zconf.attr(type=str)
    output_base_path = zconf.attr(type=str)


def lookup_and_export_model(model_type, output_base_path):
    model_class, tokenizer_class = get_model_and_tokenizer_classes(model_type)
    export_model(
        model_type=model_type,
        output_base_path=output_base_path,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
    )


def export_model(model_type, output_base_path, model_class, tokenizer_class):
    tokenizer_fol_path = os.path.join(output_base_path, "tokenizer")
    model_fol_path = os.path.join(output_base_path, "model")
    os.makedirs(tokenizer_fol_path, exist_ok=True)
    os.makedirs(model_fol_path, exist_ok=True)

    model_path = os.path.join(model_fol_path, f"{model_type}.p")
    model_config_path = os.path.join(model_fol_path, f"{model_type}.json")
    model = model_class.from_pretrained(model_type)
    torch.save(model.state_dict(), model_path)
    io.write_json(model.config.to_dict(), model_config_path)
    tokenizer = tokenizer_class.from_pretrained(model_type)
    tokenizer.save_pretrained(tokenizer_fol_path)
    config = {
        "model_type": model_type,
        "model_path": model_path,
        "model_config_path": model_config_path,
        "model_tokenizer_path": tokenizer_fol_path,
    }
    io.write_json(config, os.path.join(output_base_path, f"config.json"))


def get_model_and_tokenizer_classes(model_type):
    return CLASS_LOOKUP[model_type.split("-")[0]]


def main():
    args = RunConfiguration.run_cli()
    lookup_and_export_model(
        model_type=args.model_type,
        output_base_path=args.output_base_path,
    )


if __name__ == "__main__":
    main()
