from nlpr.shared.model_resolution import ModelArchitectures
from nlpr.shared.model_setup import ModelWrapper


def setup_adapter_model(model_type, model_class_spec,
                        config_path, tokenizer_path):
    config = model_class_spec.config_class.from_json_file(config_path)
    config.use_adapter = True

    model = model_class_spec.model_class(config)
    model_arch = ModelArchitectures.from_model_type(model_type)
    if model_arch in [ModelArchitectures.BERT]:
        if "-cased" in model_type:
            do_lower_case = False
        elif "-uncased" in model_type:
            do_lower_case = True
        else:
            raise RuntimeError(model_type)
    elif model_arch in [
            ModelArchitectures.XLNET, ModelArchitectures.XLM, ModelArchitectures.ROBERTA]:
        do_lower_case = False
    else:
        raise RuntimeError(model_type)
    print(do_lower_case)
    tokenizer = model_class_spec.tokenizer_class.from_pretrained(
        tokenizer_path, do_lower_case=do_lower_case,
    )
    model_wrapper = ModelWrapper(
        model=model,
        tokenizer=tokenizer
    )
    return model_wrapper
