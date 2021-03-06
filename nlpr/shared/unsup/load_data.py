import pyutils.io as io

import nlpr.tasks as tasks


def scrub_label(example, task):
    if task.TASK_TYPE == tasks.TaskTypes.CLASSIFICATION:
        example.label = task.LABELS[-1]
    elif task.TASK_TYPE == tasks.TaskTypes.REGRESSION:
        example.label = 0
    elif task.TASK_TYPE == tasks.TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        example.label = task.LABELS[-1]
    else:
        raise KeyError(task)


def load_single_path(path, task, prefix=None):
    raw_examples = io.read_jsonl(path)
    examples = [
        task.Example(**kwargs)
        for kwargs in raw_examples
    ]
    if prefix is not None:
        for i, example in enumerate(examples):
            example.guid = f"{prefix}-{i}"
    return examples


def load_unsup_examples_from_config(unsup_config, prefix="unsup-"):
    task = tasks.get_task_class(unsup_config["task"])
    unsup_data = {}
    if "orig" in unsup_config:
        unsup_data["orig"] = load_single_path(
            path=unsup_config["orig"],
            task=task,
            prefix=f"{prefix}-orig-",
        )
    if "aug" in unsup_config:
        unsup_data["aug"] = []
        for i, unsup_aug_path in enumerate(unsup_config["aug"]):
            unsup_data["aug"].append(load_single_path(
                path=unsup_aug_path,
                task=task,
                prefix=f"{prefix}-aug{i}-",
            ))
    return task, unsup_data


def load_unsup_examples_from_config_path(unsup_config_path, prefix="unsup-"):
    return load_unsup_examples_from_config(
        unsup_config=io.read_json(unsup_config_path),
        prefix=prefix,
    )


def load_sup_and_unsup_data(task_config_path, unsup_task_config_path):
    task = tasks.create_task_from_config_path(
        config_path=task_config_path,
        verbose=True,
    )
    unsup_task, unsup_data = \
        load_unsup_examples_from_config_path(unsup_task_config_path)
    task_data = {
        "sup": {
            "train": task.get_train_examples(),
            "val": task.get_val_examples(),
            "test": task.get_test_examples(),
        },
        "unsup": unsup_data,
    }
    return task, task_data
