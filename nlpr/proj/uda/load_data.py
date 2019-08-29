import nlpr.tasks as tasks

from pyutils.io import read_file_lines, read_json


def load_task_data(uda_config, verbose=True):
    return generic_load_task_data(uda_config, verbose=verbose)


def generic_load_task_data(uda_config, verbose=True):
    task_class = tasks.get_task_class(uda_config["task"])
    task = task_class(name=uda_config["task"], path_dict=uda_config["sup"])
    if verbose:
        print("SUP:")
        for k, v in task.path_dict.items():
            print(f"  [{k}]: {v}")

    task_data = {
        "sup": {
            "train": task.get_train_examples(),
            "val": task.get_val_examples(),
            "test": task.get_test_examples(),
        },
        "unsup": load_unsup_data(
            unsup_config=uda_config["unsup"],
            task_class=task_class,
        )
    }

    return task, task_data


def load_task_data_from_path(uda_config_path, verbose=True):
    return load_task_data(read_json(uda_config_path), verbose=verbose)


def create_examples_from_paths(path_dict, task_class, prefix="unsup-"):
    lines_dict = {
        field: read_file_lines(path)
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


def load_unsup_data(unsup_config, task_class, verbose=True):
    unsup_data = dict()
    unsup_data["orig"] = create_examples_from_paths(
        path_dict=unsup_config["orig"],
        task_class=task_class,
        prefix="orig",
    )
    unsup_data["aug"] = []
    if verbose:
        print("UNSUP:")
    for i, unsup_config in enumerate(unsup_config["aug"]):
        if verbose:
            for k, v in unsup_config.items():
                print(f"  [{k}]: {v}")
        aug_data = create_examples_from_paths(
            path_dict=unsup_config,
            task_class=task_class,
            prefix="aug",
        )
        assert len(aug_data) == len(unsup_data["orig"])
        unsup_data["aug"].append(aug_data)
    return unsup_data
