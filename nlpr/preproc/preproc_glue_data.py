import csv
import os
import tqdm

import zconf
import pyutils.io as io


GLUE_CONVERSION = {
    "cola": {
        "data": {
            "train": {"cols": {"text_a": 3, "label": 1}},
            "val": {"cols": {"text_a": 3, "label": 1},
                    "meta": {"filename": "dev"}},
            "test": {"cols": {"text_a": 1}, "meta": {"skiprows": 1}},
        },
        "dir_name": "CoLA",
    },
    "mnli": {
        "data": {
            "train": {"cols": {"text_a": 8, "text_b": 9, "label": 11},
                      "meta": {"skiprows": 1}},
            "val": {"cols": {"text_a": 8, "text_b": 9, "label": 15},
                    "meta": {"filename": "dev_matched", "skiprows": 1}},
            "val_mismatched": {"cols": {"text_a": 8, "text_b": 9, "label": 15},
                               "meta": {"filename": "dev_mismatched", "skiprows": 1}},
            "test": {"cols": {"text_a": 8, "text_b": 9},
                     "meta": {"filename": "test_matched", "skiprows": 1}},
            "test_mismatched": {"cols": {"text_a": 8, "text_b": 9},
                                "meta": {"filename": "test_mismatched", "skiprows": 1}},
        },
        "dir_name": "MNLI",
    },
    "mrpc": {
        "data": {
            "train": {"cols": {"text_a": 3, "text_b": 4, "label": 0},
                      "meta": {"skiprows": 1}},
            "val": {"cols": {"text_a": 3, "text_b": 4, "label": 0},
                    "meta": {"filename": "dev", "skiprows": 1}},
            "test": {"cols": {"text_a": 3, "text_b": 4},
                     "meta": {"skiprows": 1}},
        },
        "dir_name": "MRPC",
    },
    "qnli": {
        "data": {
            "train": {"cols": {"text_a": 1, "text_b": 2, "label": 3},
                      "meta": {"skiprows": 1}},
            "val": {"cols": {"text_a": 1, "text_b": 2, "label": 3},
                    "meta": {"filename": "dev", "skiprows": 1}},
            "test": {"cols": {"text_a": 1, "text_b": 2},
                     "meta": {"skiprows": 1}},
        },
        "dir_name": "QNLI",
    },
    "qqp": {
        "data": {
            "train": {"cols": {"text_a": 3, "text_b": 4, "label": 5},
                      "meta": {"skiprows": 1}},
            "val": {"cols": {"text_a": 3, "text_b": 4, "label": 5},
                    "meta": {"filename": "dev", "skiprows": 1}},
            "test": {"cols": {"text_a": 1, "text_b": 2},
                     "meta": {"skiprows": 1}},
        },
        "dir_name": "QQP",
    },
    "rte": {
        "data": {
            "train": {"cols": {"premise": 1, "hypothesis": 2, "label": 3},
                      "meta": {"skiprows": 1}},
            "val": {"cols": {"premise": 1, "hypothesis": 2, "label": 3},
                    "meta": {"filename": "dev", "skiprows": 1}},
            "test": {"cols": {"premise": 1, "hypothesis": 2},
                     "meta": {"skiprows": 1}},
        },
        "dir_name": "RTE",
    },
    "sst": {
        "data": {
            "train": {"cols": {"text_a": 0, "label": 1},
                      "meta": {"skiprows": 1}},
            "val": {"cols": {"text_a": 0, "label": 1},
                    "meta": {"filename": "dev", "skiprows": 1}},
            "test": {"cols": {"text_a": 1},
                     "meta": {"skiprows": 1}},
        },
        "dir_name": "SST-2",
    },
    "stsb": {
        "data": {
            "train": {"cols": {"text_a": 7, "text_b": 8, "label": 9},
                      "meta": {"skiprows": 1}},
            "val": {"cols": {"text_a": 7, "text_b": 8, "label": 9},
                    "meta": {"filename": "dev", "skiprows": 1}},
            "test": {"cols": {"text_a": 7, "text_b": 8},
                     "meta": {"skiprows": 1}},
        },
        "dir_name": "STS-B",
    },
    "wnli": {
        "data": {
            "train": {"cols": {"text_a": 1, "text_b": 2, "label": 3},
                      "meta": {"skiprows": 1}},
            "val": {"cols": {"text_a": 1, "text_b": 2, "label": 3},
                    "meta": {"filename": "dev", "skiprows": 1}},
            "test": {"cols": {"text_a": 1, "text_b": 2},
                     "meta": {"skiprows": 1}},
        },
        "dir_name": "WNLI",
    },
}


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    input_base_path = zconf.attr(type=str)
    output_base_path = zconf.attr(type=str)


def read_tsv(input_file, quotechar=None, skiprows=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        result = list(csv.reader(f, delimiter="\t", quotechar=quotechar))
    if skiprows:
        result = result[skiprows:]
    return result


def get_full_examples(task_name, input_base_path):
    task_metadata = GLUE_CONVERSION[task_name]
    all_examples = {}
    for phase, phase_config in task_metadata["data"].items():
        meta_dict = phase_config.get("meta", {})
        filename = meta_dict.get("filename", phase)
        rows = read_tsv(
            os.path.join(input_base_path, task_metadata["dir_name"], f"{filename}.tsv"),
            skiprows=meta_dict.get("skiprows"),
        )
        examples = []
        for row in rows:
            try:
                example = {}
                for col, i in phase_config["cols"].items():
                    example[col] = row[i]
                examples.append(example)
            except IndexError:
                if task_name == "qqp":
                    continue
        all_examples[phase] = examples
    return all_examples


def preprocess_all_glue_data(input_base_path, output_base_path):
    os.makedirs(output_base_path, exist_ok=True)
    os.makedirs(os.path.join(output_base_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, "configs"), exist_ok=True)
    for task_name in tqdm.tqdm(GLUE_CONVERSION):
        task_data_path = os.path.join(output_base_path, "data", task_name)
        os.makedirs(task_data_path, exist_ok=True)
        task_all_examples = get_full_examples(
            task_name=task_name,
            input_base_path=input_base_path,
        )
        config = {"task": task_name, "paths": {}}
        for phase, phase_data in task_all_examples.items():
            phase_data_path = os.path.join(task_data_path, f"{phase}.jsonl")
            io.write_jsonl(
                data=phase_data,
                path=phase_data_path,
            )
            config["paths"][phase] = phase_data_path

        io.write_json(
            data=config,
            path=os.path.join(output_base_path, "configs", f"{task_name}.json")
        )


def main():
    args = RunConfiguration.run_cli()
    preprocess_all_glue_data(
        input_base_path=args.input_base_path,
        output_base_path=args.output_base_path,
    )


if __name__ == "__main__":
    main()
