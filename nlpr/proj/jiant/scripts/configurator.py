import os

import torch

import zconf
import pyutils.io as io


class Registry:
    func_dict = {}

    @classmethod
    def register(cls, f):
        cls.func_dict[f.__name__] = f


def write_configs(config_dict, base_path, check_paths=True):
    os.makedirs(base_path, exist_ok=True)
    config_keys = [
        'task_config_path_dict', 'task_cache_config_dict', 'sampler_config',
        'global_train_config', 'task_specific_configs_dict', 'metric_aggregator_config',
        "submodels_config", "task_run_config",
    ]
    for path in config_dict["task_config_path_dict"].values():
        if check_paths:
            assert os.path.exists(path)
    for path_dict in config_dict["task_cache_config_dict"].values():
        for path in path_dict.values():
            if check_paths:
                assert os.path.exists(path)
    for config_key in config_keys:
        io.write_json(
            config_dict[config_key],
            os.path.join(base_path, f"{config_key}.json"),
        )
    io.write_json(config_dict, os.path.join(base_path, "full.json"))
    io.write_json({
        f"{config_key}_path": os.path.join(base_path, f"{config_key}.json")
        for config_key in config_keys
    }, path=os.path.join(base_path, "zz_full.json"))


def write_configs_from_full(full_config_path):
    write_configs(
        config_dict=io.read_json(full_config_path),
        base_path=os.path.split(full_config_path)[0],
    )


@Registry.register
def single_task_config(task_config_path,
                       train_batch_size,
                       task_cache_base_path=None,
                       epochs=None, max_steps=None,
                       task_cache_train_path=None,
                       task_cache_val_path=None,
                       task_cache_val_labels_path=None,
                       eval_batch_multiplier=2,
                       gradient_accumulation_steps=1,
                       eval_subset_num=500,
                       warmup_steps_proportion=0.1):
    task_config = io.read_json(os.path.expandvars(task_config_path))
    task_name = task_config["name"]

    assert (epochs is None) != (max_steps is None)

    if task_cache_train_path is None:
        task_cache_train_path = os.path.join(task_cache_base_path, "train")
    if task_cache_val_path is None:
        task_cache_val_path = os.path.join(task_cache_base_path, "val")
    if task_cache_val_labels_path is None:
        task_cache_val_labels_path = os.path.join(task_cache_base_path, "val_labels")

    if epochs is not None:
        cache_metadata = os.path.expandvars(os.path.join(task_cache_train_path, "data_args.p"))
        num_training_examples = torch.load(cache_metadata)["length"]
        max_steps = num_training_examples * epochs

    config_dict = {
        "task_config_path_dict": {
            task_name: os.path.expandvars(task_config_path),
        },
        "task_cache_config_dict": {
            task_name: {
                "train": os.path.expandvars(task_cache_train_path),
                "val": os.path.expandvars(task_cache_val_path),
                "val_labels": os.path.expandvars(task_cache_val_labels_path),
            },
        },
        "sampler_config": {
            "sampler_type": "UniformMultiTaskSampler",
        },
        "global_train_config": {
            "max_steps": max_steps,
            "warmup_steps": int(max_steps * warmup_steps_proportion),
        },
        "task_specific_configs_dict": {
            task_name: {
                "train_batch_size": train_batch_size,
                "eval_batch_size": train_batch_size * eval_batch_multiplier,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "eval_subset_num": eval_subset_num,
            },
        },
        "submodels_config": {
            "task_to_submodel_map": {
                task_name: task_name,
            },
        },
        "task_run_config": {
            "train_task_list": [task_name],
            "train_val_task_list": [task_name],
            "val_task_list": [task_name],
            "test_task_list": [task_name],
        },
        "metric_aggregator_config": {
            "metric_aggregator_type": "EqualMetricAggregator",
        },
    }
    return config_dict


@zconf.run_config
class JsonRunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    func = zconf.attr(type=str, required=True)
    path = zconf.attr(type=str, required=True)
    output_base_path = zconf.attr(type=str, required=True)


def main():
    mode, cl_args = zconf.get_mode_and_cl_args()
    if mode == "json":
        args = JsonRunConfiguration.default_run_cli(cl_args=cl_args)
        config_dict = Registry.func_dict[args.func](**io.read_json(args.path))
        write_configs(
            config_dict=config_dict,
            base_path=args.output_base_path,
        )
    else:
        raise zconf.ModeLookupError(mode)


if __name__ == "__main__":
    main()
