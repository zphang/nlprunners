import collections
import json
import os
import re
import string

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
    if isinstance(task, tasks.AnliTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.AmazonPolarityTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.BoolQTask):
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
    elif isinstance(task, tasks.ReCoRDTask):
        return RecordTaskEval.from_logits(logits, examples)
    elif isinstance(task, tasks.RteTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.SstTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.StsbTask):
        # Not actually logits
        return PearsonAndSpearmanEval.from_preds(np.squeeze(logits, axis=-1), examples)
    elif isinstance(task, tasks.WiCTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.WSCTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.YelpPolarityTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    else:
        raise KeyError(task)


def compute_task_metrics_from_classification_preds(task, preds, examples):
    # Todo: move logic to task?
    if isinstance(task, tasks.AnliTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.AmazonPolarityTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.BoolQTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.CommitmentBankTask):
        return CommitmentBankEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.ColaTask):
        return MccEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.CopaTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.IMDBTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.MnliTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.MrpcTask):
        return AccAndF1Eval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.MultiRCTask):
        return MultiRCEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.QnliTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.QqpTask):
        return AccAndF1Eval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.RteTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.SstTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.StsbTask):
        raise RuntimeError("Not supported for regression")
    elif isinstance(task, tasks.WiCTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.WSCTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.YelpPolarityTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    else:
        raise KeyError(task)


def compute_task_metrics_from_classification_preds_and_labels(task, preds, labels):
    # Todo: move logic to task?
    if isinstance(task, tasks.AnliTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.AmazonPolarityTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.BoolQTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.CommitmentBankTask):
        return CommitmentBankEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.ColaTask):
        return MccEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.CopaTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.IMDBTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.MnliTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.MrpcTask):
        return AccAndF1Eval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.MultiRCTask):
        # labels is a lists of dicts
        return MultiRCEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.QnliTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.QqpTask):
        return AccAndF1Eval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.RteTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.SstTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.StsbTask):
        raise RuntimeError("Not supported for regression")
    elif isinstance(task, tasks.WiCTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.WSCTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.YelpPolarityTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    else:
        raise KeyError(task)


def compute_task_metrics_from_classification_logits_and_labels(task, logits, labels):
    # Todo: move logic to task?
    if isinstance(task, tasks.AnliTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.AmazonPolarityTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.BoolQTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.CommitmentBankTask):
        return CommitmentBankEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.ColaTask):
        return MccEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.CopaTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.IMDBTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.MnliTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.MrpcTask):
        return AccAndF1Eval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.MultiRCTask):
        # labels is a lists of dicts
        return MultiRCEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.QnliTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.QqpTask):
        return AccAndF1Eval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.RteTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.SstTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.StsbTask):
        raise RuntimeError("Not supported for regression")
    elif isinstance(task, tasks.WiCTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.WSCTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.YelpPolarityTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    else:
        raise KeyError(task)


def get_labels_from_examples(task, examples):
    # Todo: move logic to task?
    if isinstance(task, tasks.AnliTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.AmazonPolarityTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.BoolQTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.CommitmentBankTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.ColaTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.CopaTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.IMDBTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.MnliTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.MrpcTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.MultiRCTask):
        # labels is a lists of dicts
        return MultiRCEval.get_labels_from_examples(task=task, examples=examples)
    elif isinstance(task, tasks.QnliTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.QqpTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.RteTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.SstTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.StsbTask):
        return get_label_vals(examples=examples)
    elif isinstance(task, tasks.WiCTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.WSCTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.YelpPolarityTask):
        return get_label_ids(task=task, examples=examples)
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
        labels = cls.get_labels_from_examples(task=task, examples=examples)
        return cls.from_preds_and_labels(preds=preds, labels=labels)

    @classmethod
    def from_preds_and_labels(cls, preds, labels):
        df = pd.DataFrame(labels)
        assert "label_values" in df.columns
        assert "question_ids" in df.columns
        df["preds"] = preds
        exact_match = df \
            .groupby("question_ids") \
            .apply(lambda _: (_["preds"] == _["label_values"]).all()) \
            .mean()
        exact_match = float(exact_match)
        f1 = f1_score(y_true=df["label_values"], y_pred=df["preds"])
        return Metrics(
            major=mean(exact_match, f1),
            minor={"em": exact_match, "f1": f1},
        )

    @classmethod
    def get_labels_from_examples(cls, task, examples):
        label_values = get_label_ids(examples=examples, task=task)
        question_ids = np.array([example.question_id for example in examples])
        assert len(label_values) == len(question_ids)
        return [
            {"label_values": lab, "question_ids": qid}
            for lab, qid in zip(label_values, question_ids)
        ]


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


class RecordTaskEval(BaseEvaluation):
    @classmethod
    def from_logits(cls, logits, examples):
        psg_qns_idx_dict = {}
        for i, example in examples:
            psq_qns_idx = example.passage_idx, example.question_idx
            if psq_qns_idx not in psg_qns_idx_dict:
                psg_qns_idx_dict[psq_qns_idx] = []
            psg_qns_idx_dict[psq_qns_idx].append(i)

        f1_ls = []
        em_ls = []

        for psq_qns_idx, example_indices in psg_qns_idx_dict:
            # answer_dict should be same across all examples with the same psq_qns_idx
            relevant_examples = [examples[i] for i in example_indices]
            golds = list(relevant_examples[0].answers_dict.values())
            psg_qns_logits = logits[example_indices]
            psg_qns_pred = np.argmax(psg_qns_logits[:, 1])  # Take argmax over positive preds
            pred_ans = relevant_examples[psg_qns_pred].entity_str

            # F1
            f1 = cls.metric_max_over_ground_truths(cls.f1_score, pred_ans, golds)
            f1_ls.append(f1)

            # EM
            em = cls.metric_max_over_ground_truths(cls.exact_match_score, pred_ans, golds)
            em_ls.append(em)

        em = sum(em_ls) / len(em_ls)
        f1 = sum(f1_ls) / len(f1_ls)
        minor = {
            "em": em,
            "f1": f1,
            "f1_em": (f1 + em) / 2,
        }
        return Metrics(
            major=minor["f1_em"],
            minor=minor,
        )

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

    @classmethod
    def normalize_answer(cls, s):
        """Lower text and remove punctuation, articles and extra whitespace.
        From official ReCoRD eval script """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def f1_score(cls, prediction, ground_truth):
        """ Compute normalized token level F1
        From official ReCoRD eval script """
        prediction_tokens = cls.normalize_answer(prediction).split()
        ground_truth_tokens = cls.normalize_answer(ground_truth).split()
        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @classmethod
    def exact_match_score(cls, prediction, ground_truth):
        """ Compute normalized exact match
        From official ReCoRD eval script """
        return cls.normalize_answer(prediction) == cls.normalize_answer(ground_truth)

    @classmethod
    def metric_max_over_ground_truths(cls, metric_fn, prediction, ground_truths):
        """ Compute max metric between prediction and each ground truth.
        From official ReCoRD eval script """
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)


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
    results_to_write = {}
    if "loss" in results:
        results_to_write["loss"] = results["loss"]
    if "metrics" in results:
        results_to_write["metrics"] = results["metrics"].asdict()
    assert results_to_write
    metrics_str = json.dumps(results_to_write, indent=2)
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
