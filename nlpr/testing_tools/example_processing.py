import glob
import numpy as np
import os
import tqdm

import torch

import pyutils.io as io

import nlpr.shared.model_resolution as model_resolution
import nlpr.shared.model_setup as model_setup
import nlpr.tasks as tasks
from nlpr.tasks.lib.shared import TaskTypes


TASK_EXCLUSION_LS = ["wic", "wsc"]
MODEL_NAME_LS = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]


def random_choose(ls, num, rng=None, replace=True):
    if rng is None:
        rng = np.random
    return [ls[i] for i in rng.choice(len(ls), num, replace=replace)]


def write_truncated_config(config_base_path, output_base_path):
    path_ls = sorted(glob.glob(os.path.join(config_base_path, "*/base_config.json")))
    for path in tqdm.tqdm(path_ls):
        config = io.read_json(path)
        data = io.read_jsonl(io.read_json(path)["paths"]["train"])
        task_name = config["task"]
        new_data_path = os.path.join(output_base_path, "task_data", f"{task_name}.json")
        new_data = random_choose(data, 20, np.random.RandomState(1111))
        new_config = {
            "task": task_name,
            "paths": {"train": new_data_path}
        }
        io.write_jsonl(new_data, new_data_path)
        io.write_json(new_config, os.path.join(output_base_path, "task_configs", f"{task_name}.json"))


def get_tokenizer_and_feat_spec(model_config):
    model_class_spec = model_resolution.resolve_model_setup_classes(
        model_type=model_config["model_type"],
        task_type=TaskTypes.CLASSIFICATION,  # Doesn't matter in this case
    )
    feat_spec = model_resolution.build_featurization_spec(
        model_type=model_config["model_type"],
        max_seq_length=128,
    )
    tokenizer = model_setup.get_tokenizer(
        model_type=model_config["model_type"],
        model_class_spec=model_class_spec,
        tokenizer_path=model_config["model_tokenizer_path"],
    )
    return tokenizer, feat_spec


def get_tokenized_featurized(examples, tokenizer, feat_spec):
    tokenized_example_ls = [
        example.tokenize(tokenizer)
        for example in examples
    ]
    featurized_example_ls = [
        tokenized_example.featurize(tokenizer, feat_spec)
        for tokenized_example in tokenized_example_ls
    ]
    return {
        "tokenized": [x.asdict() for x in tokenized_example_ls],
        "featurized": [x.asdict() for x in featurized_example_ls],
    }


def write_out(model_config_base_path, task_config_base_path, output_base_path, metadata=None):
    task_config_path_dict = {
        os.path.split(path)[-1].replace(".json", ""): path
        for path in glob.glob(os.path.join(task_config_base_path, "*.json"))
    }
    for model_name in tqdm.tqdm(MODEL_NAME_LS):
        os.makedirs(os.path.join(output_base_path, model_name), exist_ok=True)
        model_config = io.read_json(os.path.join(model_config_base_path, f"{model_name}.json"))
        tokenizer, feat_spec = get_tokenizer_and_feat_spec(model_config)
        for task_name, task_config_path in tqdm.tqdm(task_config_path_dict.items(), total=len(task_config_path_dict)):
            if task_name in TASK_EXCLUSION_LS:
                continue
            task = tasks.create_task_from_config_path(config_path=task_config_path)
            examples = task.get_train_examples()
            output = get_tokenized_featurized(
                examples=examples,
                tokenizer=tokenizer,
                feat_spec=feat_spec,
            )
            torch.save(
                output,
                os.path.join(output_base_path, model_name, f"{task_name}.p"),
            )
    io.write_json(
        metadata,
        os.path.join(output_base_path, "metadata")
    )


def check_equal(ex1, ex2):
    try:
        assert isinstance(ex1, dict)
        assert isinstance(ex2, dict)
        assert sorted(list(ex1.keys())) == sorted(list(ex2.keys()))
        for k in ex1.keys():
            v1, v2 = ex1[k], ex2[k]
            assert isinstance(v2, type(v1))
            if isinstance(v1, (str, int, list, float)):
                assert v1 == v2
            else:
                raise RuntimeError()
        return True
    except AssertionError:
        return False


def check_data_equal(data1, data2):
    for ex1, ex2 in zip(data1["tokenized"], data2["tokenized"]):
        if not check_equal(ex1, ex2):
            return False
    for ex1, ex2 in zip(data1["featurized"], data2["featurized"]):
        if not check_equal(ex1, ex2):
            return False
    return True


def run_checks(base_path_1, base_path_2):
    failed = 0
    total = 0
    for model_name in MODEL_NAME_LS:
        model_path_1 = os.path.join(base_path_1, model_name)
        model_path_2 = os.path.join(base_path_2, model_name)
        task_name_ls = [
            os.path.split(x)[-1].replace(".p", "")
            for x in sorted(glob.glob(os.path.join(model_path_1, "*.p")))
        ]
        print(model_name)
        for task_name in task_name_ls:
            total += 1
            data1 = torch.load(os.path.join(model_path_1, f"{task_name}.p"))
            data2 = torch.load(os.path.join(model_path_2, f"{task_name}.p"))
            if check_data_equal(data1, data2):
                print(f"    {task_name}:", "OK")
            else:
                print(f"    {task_name}:", "FAILED")
                failed += 1
    if failed:
        print(f"{failed}/{total} FAILED")
    else:
        print(f"{total}/{total} PASSED")
