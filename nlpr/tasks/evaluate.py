import json
import os

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from typing import Dict

import nlpr.tasks as tasks
from nlpr.shared.pycore import ExtendedDataClassMixin


@dataclass
class Metrics(ExtendedDataClassMixin):
    major: float
    minor: Dict


class BaseEvaluation:
    pass


def compute_task_metrics(task, logits, examples):
    # Todo: move logic to task?
    if isinstance(task, tasks.BoolQTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.CommitmentBankTask):
        return CommitmentBankEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.ColaTask):
        return MccEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.CopaTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.IMDBTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.MnliTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.MrpcTask):
        return AccAndF1Eval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.MultiRCTask):
        return MultiRCEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.QnliTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.QqpTask):
        return AccAndF1Eval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.RteTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.SstTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.StsbTask):
        # Not actually logits
        return PearsonAndSpearmanEval.from_preds(logits, examples)
    elif isinstance(task, tasks.WiCTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.WSCTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    else:
        raise KeyError(task)


class SimpleAccuracyEval(BaseEvaluation):
    @classmethod
    def from_logits(cls, task, logits, examples):
        return cls.from_preds(task=task, preds=get_preds(logits), examples=examples)

    @classmethod
    def from_preds(cls, task, preds, examples):
        labels = get_label_ids(examples=examples, task=task)
        return cls.from_preds_and_labels(preds=preds, labels=labels)

    @classmethod
    def from_preds_and_labels(cls, preds, labels):
        acc = float((preds == labels).mean())
        return Metrics(
            major=acc,
            minor={"acc": acc},
        )


class CommitmentBankEval(BaseEvaluation):
    @classmethod
    def from_logits(cls, task, logits, examples):
        return cls.from_preds(task=task, preds=get_preds(logits), examples=examples)

    @classmethod
    def from_preds(cls, task, preds, examples):
        labels = get_label_ids(examples=examples, task=task)
        return cls.from_preds_and_labels(preds=preds, labels=labels)

    @classmethod
    def from_preds_and_labels(cls, preds, labels):
        acc = float((preds == labels).mean())
        f11 = f1_score(y_true=labels == 0, y_pred=preds == 0)
        f12 = f1_score(y_true=labels == 1, y_pred=preds == 1)
        f13 = f1_score(y_true=labels == 2, y_pred=preds == 2)
        avg_f1 = mean(f11, f12, f13)
        return Metrics(
            major=mean(acc, avg_f1),
            minor={"acc": acc, "avg_f1": avg_f1,
                   "f11": f11, "f12": f12, "f13": f13}
        )


class MultiRCEval(BaseEvaluation):
    @classmethod
    def from_logits(cls, task, logits, examples):
        return cls.from_preds(task=task, preds=get_preds(logits), examples=examples)

    @classmethod
    def from_preds(cls, task, preds, examples):
        labels = get_label_ids(examples=examples, task=task)
        question_ids = np.array([example.question_id for example in examples])
        return cls.from_preds_and_labels(preds=preds, labels=labels, question_ids=question_ids)

    @classmethod
    def from_preds_and_labels(cls, preds, labels, question_ids):
        df = pd.DataFrame({
            "preds": preds,
            "labels": labels,
            "question_ids": question_ids,
        })
        exact_match = df \
            .groupby("question_ids") \
            .apply(lambda _: (_["preds"] == _["labels"]).all()) \
            .mean()
        exact_match = float(exact_match)
        f1 = f1_score(y_true=df["labels"], y_pred=df["preds"])
        return Metrics(
            major=mean(exact_match, f1),
            minor={"em": exact_match, "f1": f1},
        )


class AccAndF1Eval(BaseEvaluation):
    @classmethod
    def from_logits(cls, task, logits, examples):
        return cls.from_preds(task=task, preds=get_preds(logits), examples=examples)

    @classmethod
    def from_preds(cls, task, preds, examples):
        labels = get_label_ids(examples=examples, task=task)
        return cls.from_preds_and_labels(preds=preds, labels=labels)

    @classmethod
    def from_preds_and_labels(cls, preds, labels):
        acc = float((preds == labels).mean())
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


class MccEval(BaseEvaluation):
    @classmethod
    def from_logits(cls, task, logits, examples):
        return cls.from_preds(task=task, preds=get_preds(logits), examples=examples)

    @classmethod
    def from_preds(cls, task, preds, examples):
        labels = get_label_ids(examples=examples, task=task)
        return cls.from_preds_and_labels(preds=preds, labels=labels)

    @classmethod
    def from_preds_and_labels(cls, preds, labels):
        mcc = matthews_corrcoef(labels, preds)
        return Metrics(
            major=mcc,
            minor={"mcc": mcc},
        )


class PearsonAndSpearmanEval(BaseEvaluation):

    @classmethod
    def from_preds(cls, preds, examples):
        true_values = get_label_vals(examples)
        return cls.from_preds_and_labels(preds=preds, true_values=true_values)

    @classmethod
    def from_preds_and_labels(cls, preds, true_values):
        pearson_corr = float(pearsonr(preds, true_values)[0])
        spearman_corr = float(spearmanr(preds, true_values)[0])
        minor = {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }
        return Metrics(
            major=minor["corr"],
            minor=minor,
        )


def get_preds(logits):
    return np.argmax(logits, axis=1)


def get_label_ids(examples, task):
    return np.array([task.LABEL_BIMAP.a[example.label] for example in examples])


def get_label_vals(examples):
    return np.array([example.label for example in examples])


def mean(*args) -> float:
    return float(np.mean(args))


def write_metrics(results, output_path, verbose=True):
    metrics_str = json.dumps(
        {"loss": results["loss"], "metrics": results["metrics"].asdict()},
        indent=2,
    )
    if verbose:
        print(metrics_str)
    with open(output_path, "w") as f:
        f.write(metrics_str)


def write_preds(logits, output_path):
    df = pd.DataFrame(logits)
    df.to_csv(output_path, header=False, index=False)


def write_val_results(results, output_dir, verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    write_preds(
        logits=results["logits"],
        output_path=os.path.join(output_dir, "val_preds.csv"),
    )
    write_metrics(
        results=results,
        output_path=os.path.join(output_dir, "val_metrics.json"),
        verbose=verbose,
    )
