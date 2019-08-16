from nlpr.tasks import MnliTask, BoolQTask

from pyutils.io import read_file_lines, read_json


def load_task_data(uda_config):
    if uda_config["task"] == "mnli":
        return load_mnli_data(uda_config)
    if uda_config["task"] == "boolq":
        return load_boolq_data(uda_config)
    else:
        raise KeyError(uda_config["task"])


def load_task_data_from_path(uda_config_path):
    return load_task_data(read_json(uda_config_path))


def load_boolq_data(uda_config, verbose=True):
    task = BoolQTask("boolq", path_dict=uda_config["sup"])
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
        "unsup": {
            "orig": {},
            "aug": [],
        },
    }

    def _load_boolq_from_files(question_path, passage_path, prefix="unsup-"):
        unsup_examples = []
        question_lines = read_file_lines(question_path)
        passage_lines = read_file_lines(passage_path)
        for j, (question, passage) in \
                enumerate(zip(question_lines, passage_lines)):
            unsup_examples.append(BoolQTask.Example(
                guid=f"{prefix}-{j}",
                input_question=question.strip(),
                input_passage=passage.strip(),
                label=BoolQTask.LABELS[-1],
            ))
        return unsup_examples

    task_data["unsup"]["orig"] = _load_boolq_from_files(
        question_path=uda_config["unsup"]["orig"]["question"],
        passage_path=uda_config["unsup"]["orig"]["passage"],
    )
    if verbose:
        print("UNSUP:")
    for i, unsup_config in enumerate(uda_config["unsup"]["aug"]):
        if verbose:
            print(f"  [QUE]: {unsup_config['question']}")
            print(f"  [PAS]: {unsup_config['passage']}")
        aug_data = _load_boolq_from_files(
            question_path=unsup_config["question"],
            passage_path=unsup_config["passage"],
            prefix=f"aug-{i}",
        )
        assert len(aug_data) == len(task_data["unsup"]["orig"])
        task_data["unsup"]["aug"].append(aug_data)

    return task, task_data


def load_mnli_data(uda_config, verbose=True):
    task = MnliTask("MNLI", path_dict=uda_config["sup"])
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
        "unsup": {
            "orig": {},
            "aug": [],
        },
    }

    def _load_mnli_from_files(premise_path, hypothesis_path, prefix="unsup-"):
        unsup_examples = []
        premise_lines = read_file_lines(premise_path)
        hypothesis_lines = read_file_lines(hypothesis_path)
        for j, (premise, hypothesis) in \
                enumerate(zip(premise_lines, hypothesis_lines)):
            unsup_examples.append(MnliTask.Example(
                guid=f"{prefix}-{j}",
                input_premise=premise.strip(),
                input_hypothesis=hypothesis.strip(),
                label=MnliTask.LABELS[-1],
            ))
        return unsup_examples

    task_data["unsup"]["orig"] = _load_mnli_from_files(
        premise_path=uda_config["unsup"]["orig"]["premise"],
        hypothesis_path=uda_config["unsup"]["orig"]["hypothesis"],
    )
    if verbose:
        print("UNSUP:")
    for i, unsup_config in enumerate(uda_config["unsup"]["aug"]):
        if verbose:
            print(f"  [PRE]: {unsup_config['premise']}")
            print(f"  [HYP]: {unsup_config['hypothesis']}")
        aug_data = _load_mnli_from_files(
            premise_path=unsup_config["premise"],
            hypothesis_path=unsup_config["hypothesis"],
            prefix=f"aug-{i}",
        )
        assert len(aug_data) == len(task_data["unsup"]["orig"])
        task_data["unsup"]["aug"].append(aug_data)

    return task, task_data
