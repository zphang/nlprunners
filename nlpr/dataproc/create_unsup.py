import os

import nlpr.tasks as tasks
import nlpr.shared.unsup.load_data as unsup_load_data

import pyutils.io as io


GLOBAL_EXCLUSION = ["guid", "label"]
TASK_EXCLUSION_DICT = {
    "wic": ["start1", "start2", "end1", "end2", "word"]
}


def create_input_texts_and_configs(config_path, config_output_path, output_base_path, verbose=True):
    task = tasks.create_task_from_config_path(config_path)
    train_examples = task.get_train_examples()
    unsup_config = {
        "task": task.name,
        "orig": None,
        "aug": []
    }
    for example in train_examples:
        unsup_load_data.scrub_label(example, task)
    if verbose:
        print(task.name)

    # Write scrubbed examples
    orig_data_path = os.path.join(output_base_path, task.name, "orig", f"train.unsup.jsonl")
    io.create_containing_folder(orig_data_path)
    io.write_jsonl(
        [example.asdict() for example in train_examples],
        orig_data_path,
    )
    unsup_config["orig"] = orig_data_path

    for field_name, field in task.Example.__dataclass_fields__.items():
        if field.type != str:
            continue
        if field_name in GLOBAL_EXCLUSION:
            continue
        if field_name in TASK_EXCLUSION_DICT.get(task.name, []):
            continue

        if verbose:
            print(f"  {field_name}")

        # Write text files
        orig_field_txt_path = os.path.join(
            output_base_path, task.name, "orig", f"train_{field_name}.txt")
        io.create_containing_folder(orig_field_txt_path)
        io.write_file(
            "\n".join([getattr(example, field_name).strip() for example in train_examples]),
            orig_field_txt_path,
        )

        io.create_containing_folder(config_output_path)
        io.write_json(
            unsup_config, config_output_path
        )
