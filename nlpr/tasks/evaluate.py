import json
import os

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from typing import Dict

import nlpr.tasks as tasks
from nlpr.shared.pycore import ExtendedDataClassMixin


@dataclass
class Metrics(ExtendedDataClassMixin):
    major: float
    minor: Dict


def compute_task_metrics(task, logits, examples):
    # Todo: move logic to task?
    if isinstance(task, tasks.CommitmentBankTask):
        preds = get_preds(logits)
        labels = get_label_ids(examples, task)
        acc = (preds == labels).mean()
        f11 = f1_score(y_true=labels == 0, y_pred=preds == 0)
        f12 = f1_score(y_true=labels == 1, y_pred=preds == 1)
        f13 = f1_score(y_true=labels == 2, y_pred=preds == 2)
        avg_f1 = mean(f11, f12, f13)
        return Metrics(
            major=mean(acc, avg_f1),
            minor={"acc": acc, "avg_f1": avg_f1,
                   "f11": f11, "f12": f12, "f13": f13}
        )
    elif isinstance(task, tasks.CopaTask):
        return simple_accuracy(task, logits, examples)
    elif isinstance(task, tasks.MultiRCTask):
        df = pd.DataFrame({
            "preds": get_preds(logits),
            "labels": get_label_ids(examples, task),
            "question_ids": np.array([example.question_id for example in examples]),
        })
        exact_match = df \
            .groupby("question_ids") \
            .apply(lambda _: (_["preds"] == _["labels"]).all()) \
            .mean()
        exact_match = float(exact_match)
        f1 = f1_score(y_true=df["labels"], y_pred=df["preds"])
        return Metrics(
            major=mean(exact_match, f1),
            minor={"em": exact_match, "f1": f1}
        )

    elif isinstance(task, tasks.RteTask):
        return simple_accuracy(task, logits, examples)
    elif isinstance(task, tasks.WSCTask):
        return simple_accuracy(task, logits, examples)
    elif isinstance(task, tasks.WiCTask):
        return simple_accuracy(task, logits, examples)
    elif isinstance(task, tasks.MnliTask):
        return simple_accuracy(task, logits, examples)
    elif isinstance(task, tasks.MrpcTask):
        return acc_and_f1(task, logits, examples)
    elif isinstance(task, tasks.IMDBTask):
        return simple_accuracy(task, logits, examples)
    else:
        raise KeyError(task)


def simple_accuracy(task, logits, examples):
    preds = get_preds(logits)
    labels = get_label_ids(examples, task)
    acc = float((preds == labels).mean())
    return Metrics(
        major=acc,
        minor={"acc": acc}
    )


def acc_and_f1(task, logits, examples):
    preds = get_preds(logits)
    labels = get_label_ids(examples, task)
    acc = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds)
    minor = {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }
    return Metrics(
        major=minor["acc_and_f1"],
        minor=minor,
    )


def get_preds(logits):
    return np.argmax(logits, axis=1)


def get_label_ids(examples, task):
    return np.array([task.LABEL_BIMAP.a[example.label] for example in examples])


def mean(*args) -> float:
    return float(np.mean(args))


def write_val_results(results, output_dir, verbose=True):
    df = pd.DataFrame(results["logits"])
    df.to_csv(os.path.join(output_dir, "val_preds.csv"), header=False, index=False)
    metrics_str = json.dumps(
        {"loss": results["loss"], "metrics": results["metrics"].asdict()},
        indent=2,
    )
    if verbose:
        print(metrics_str)
    with open(os.path.join(output_dir, "val_metrics.json"), "w") as f:
        f.write(metrics_str)


def write_test_preds(logits, output_path):
    df = pd.DataFrame(logits)
    df.to_csv(output_path, header=False, index=False)