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
    return unsup_data


"""
def create_examples_from_paths(path_dict, task_class, prefix="unsup-"):
    lines_dict = {
        field: io.read_file_lines(path)
        for field, path in path_dict.items()
    }
    length = len(list(lines_dict.values())[0])
    unsup_examples = []
    for i in range(length):
        examples_fields_dict = {
            "guid": f"{prefix}-{i}",
            "label": task_class.LABELS[-1],
        }
        for key, lines in lines_dict.items():
            examples_fields_dict[f"input_{key}"] = lines[i].strip()
        unsup_examples.append(task_class.Example(**examples_fields_dict))
    return unsup_examples
"""
