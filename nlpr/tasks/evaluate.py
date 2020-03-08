import collections
import json
import os
import re
import string
import torch

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from typing import Dict

import nlpr.shared.preprocessing
import nlpr.tasks as tasks
from nlpr.shared.pycore import ExtendedDataClassMixin
import nlpr.tasks.lib.templates.squad_style as squad_lib
import nlpr.shared.model_resolution as model_resolution


@dataclass
class Metrics(ExtendedDataClassMixin):
    major: float
    minor: Dict


class BaseEvaluation:
    pass


def compute_task_metrics_for_validation(task, logits, loss, labels, tokenizer):
    if isinstance(task, tasks.MLMTask):
        perplexity = np.exp(loss)
        return Metrics(
            major=perplexity,
            minor={
                "perplexity": perplexity,
            }
        )
    else:
        return compute_task_metrics_from_classification_logits_and_labels(
            task=task,
            logits=logits,
            labels=labels,
            tokenizer=tokenizer,
        )


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
    elif isinstance(task, (
                tasks.CommonsenseQATask,
                tasks.CosmosQATask,
                tasks.SWAGTask,
                tasks.HellaSwagTask,
                tasks.SocialIQATask,
            )):
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
    elif isinstance(task, tasks.SciTailTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.SnliTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.SstTask):
        return SimpleAccuracyEval.from_logits(task, logits, examples)
    elif isinstance(task, tasks.StsbTask):
        return PearsonAndSpearmanEval.from_preds(PearsonAndSpearmanEval.get_preds(logits), examples)
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
    elif isinstance(task, (
                tasks.CommonsenseQATask,
                tasks.CosmosQATask,
                tasks.SWAGTask,
                tasks.HellaSwagTask,
                tasks.SocialIQATask,
            )):
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
    elif isinstance(task, tasks.SciTailTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.SnliTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.SstTask):
        return SimpleAccuracyEval.from_preds(task, preds, examples)
    elif isinstance(task, tasks.StsbTask):
        return PearsonAndSpearmanEval.from_preds(preds, examples)
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
    elif isinstance(task, tasks.CCGTask):
        return CCGEval.from_logits_and_labels(preds, labels)
    elif isinstance(task, tasks.CommitmentBankTask):
        return CommitmentBankEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.ColaTask):
        return MccEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.CopaTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, (
                tasks.CommonsenseQATask,
                tasks.CosmosQATask,
                tasks.SWAGTask,
                tasks.HellaSwagTask,
                tasks.SocialIQATask,
            )):
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
    elif isinstance(task, tasks.SciTailTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.SnliTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.SstTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.StsbTask):
        return PearsonAndSpearmanEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.WiCTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.WSCTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    elif isinstance(task, tasks.YelpPolarityTask):
        return SimpleAccuracyEval.from_preds_and_labels(preds, labels)
    else:
        raise KeyError(task)


def compute_task_metrics_from_classification_logits_and_labels(
        task, logits, labels, tokenizer):
    # Todo: move logic to task?
    if isinstance(task, tasks.AnliTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.AmazonPolarityTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.BoolQTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.CCGTask):
        return CCGEval.from_logits_and_labels(CCGEval.get_preds(logits), labels)
    elif isinstance(task, tasks.CommitmentBankTask):
        return CommitmentBankEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.ColaTask):
        return MccEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.CopaTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, (
                tasks.CommonsenseQATask,
                tasks.CosmosQATask,
                tasks.SWAGTask,
                tasks.HellaSwagTask,
                tasks.SocialIQATask,
            )):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.IMDBTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.MnliTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.MrpcTask):
        return AccAndF1Eval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.MultiQATask):
        return SQuADEval.from_logits_and_labels(
            task=task, logits=logits, labels=labels, tokenizer=tokenizer,
        )
    elif isinstance(task, tasks.MultiRCTask):
        # labels is a lists of dicts
        return MultiRCEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.QnliTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.QqpTask):
        return AccAndF1Eval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.RteTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.SciTailTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.SnliTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.SquadTask):
        return SQuADEval.from_logits_and_labels(
            task=task, logits=logits, labels=labels, tokenizer=tokenizer,
        )
    elif isinstance(task, tasks.SstTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.StsbTask):
        return PearsonAndSpearmanEval.from_preds_and_labels(
            preds=PearsonAndSpearmanEval.get_preds(logits),
            true_values=labels,
        )
    elif isinstance(task, tasks.WiCTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.WSCTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    elif isinstance(task, tasks.YelpPolarityTask):
        return SimpleAccuracyEval.from_preds_and_labels(get_preds(logits), labels)
    else:
        raise KeyError(task)


def get_labels_from_cache(task, cache):
    if isinstance(task, (
                tasks.AnliTask,
                tasks.AmazonPolarityTask,
                tasks.BoolQTask,
                tasks.CommitmentBankTask,
                tasks.ColaTask,
                tasks.CopaTask,
                tasks.IMDBTask,
                tasks.MnliTask,
                tasks.MrpcTask,
                tasks.QnliTask,
                tasks.QqpTask,
                tasks.RteTask,
                tasks.SciTailTask,
                tasks.SnliTask,
                tasks.SstTask,
                tasks.WiCTask,
                tasks.WSCTask,
                tasks.YelpPolarityTask,
            )):
        return get_label_ids_from_cache(task=task, cache=cache)
    elif isinstance(task, tasks.StsbTask):
        return get_label_vals_from_cache(cache=cache)
    elif isinstance(task, tasks.CCGTask):
        return CCGEval.get_labels_from_cache(cache=cache)
    elif isinstance(task, (
                tasks.CommonsenseQATask,
                tasks.CosmosQATask,
                tasks.SWAGTask,
                tasks.HellaSwagTask,
                tasks.SocialIQATask,
            )):
        return get_multiple_choice_labels_from_cache(task=task, cache=cache)
    elif isinstance(task, tasks.MLMTask):
        # Labels come from inputs
        return [None]
    elif isinstance(task, (
                tasks.MultiQATask,
                tasks.SquadTask
            )):
        return SQuADEval.get_labels_from_cache(cache=cache)
    elif isinstance(task, tasks.MultiRCTask):
        # labels is a lists of dicts
        return MultiRCEval.get_label_from_cache(cache=cache)
    else:
        raise KeyError(task)


def get_labels_from_examples(task, examples, tokenizer, feat_spec, phase):
    # Todo: move logic to task?
    if isinstance(task, tasks.AnliTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.AmazonPolarityTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.BoolQTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.CCGTask):
        return CCGEval.get_labels_from_examples(
            examples=examples,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            phase=phase,
        )
    elif isinstance(task, tasks.CommitmentBankTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.ColaTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.CopaTask):
        return get_multiple_choice_label_ids(task=task, examples=examples)
    elif isinstance(task, (
                tasks.CommonsenseQATask,
                tasks.CosmosQATask,
                tasks.SWAGTask,
                tasks.HellaSwagTask,
                tasks.SocialIQATask,
            )):
        return get_multiple_choice_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.IMDBTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.MLMTask):
        return [None] * len(examples)
    elif isinstance(task, tasks.MnliTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.MrpcTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.MultiQATask):
        return SQuADEval.get_labels_from_examples(
            examples=examples,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            phase=phase,
        )
    elif isinstance(task, tasks.MultiRCTask):
        # labels is a lists of dicts
        return MultiRCEval.get_labels_from_examples(task=task, examples=examples)
    elif isinstance(task, tasks.QnliTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.QqpTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.RteTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.SciTailTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.SnliTask):
        return get_label_ids(task=task, examples=examples)
    elif isinstance(task, tasks.SquadTask):
        return SQuADEval.get_labels_from_examples(
            examples=examples,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            phase=phase,
        )
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
        labels = np.array(labels)
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

    @classmethod
    def get_label_from_cache(cls, cache):
        label_values = []
        question_ids = []
        for datum in cache.iter_all():
            label_values.append(datum["data_row"].label_id)
            question_ids.append(datum["data_row"].question_id)
        label_values = np.array(label_values)
        question_ids = np.array(question_ids)
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
        labels = np.array(labels)
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
        labels = np.array(labels)
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
    def get_preds(cls, logits):
        return np.squeeze(logits, axis=-1)

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


class SQuADEval(BaseEvaluation):

    @classmethod
    def from_logits_and_labels(cls, task, tokenizer, logits, labels):
        results, predictions = squad_lib.compute_predictions_logits_v3(
            data_rows=labels,
            logits=logits,
            n_best_size=task.n_best_size,
            max_answer_length=task.max_answer_length,
            do_lower_case=model_resolution.resolve_is_lower_case(tokenizer),
            version_2_with_negative=task.version_2_with_negative,
            null_score_diff_threshold=task.null_score_diff_threshold,
            tokenizer=tokenizer,
        )
        return Metrics(
            major=results["f1"],
            minor=results,
        )

    @classmethod
    def get_labels_from_examples(cls, examples, feat_spec, tokenizer, phase):
        dataset = nlpr.shared.preprocessing.convert_examples_to_dataset(
            examples=examples,
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            phase=phase,
            verbose=True,
        )
        return [cls.get_label_from_data_row(datum["data_row"]) for datum in dataset.data]

    @classmethod
    def get_label_from_data_row(cls, data_row):
        return squad_lib.PartialDataRow.from_data_row(data_row)

    @classmethod
    def get_labels_from_cache(cls, cache):
        return [cls.get_label_from_data_row(datum["data_row"]) for datum in cache.iter_all()]


class CCGEval(BaseEvaluation):
    # Todo: Generalize to tagging accuracy eval

    @classmethod
    def get_preds(cls, logits):
        return np.argmax(logits, axis=-1)

    @classmethod
    def from_logits_and_labels(cls, preds, labels):
        label_ids = np.stack([row["label_ids"] for row in labels])
        label_mask = np.stack([row["label_mask"] for row in labels])

        # Account for smart-truncate
        assert (label_mask[:, preds.shape[-1]:] == 0).all()
        label_ids = label_ids[:, :preds.shape[-1]]
        label_mask = label_mask[:, :preds.shape[-1]]

        bool_mask = label_mask.reshape(-1).astype(bool)
        flat_preds = preds.reshape(-1)[bool_mask]
        flat_labels = label_ids.reshape(-1)[bool_mask]
        return SimpleAccuracyEval.from_preds_and_labels(preds=flat_preds, labels=flat_labels)

    @classmethod
    def get_labels_from_examples(cls, examples, feat_spec, tokenizer, phase):
        dataset = nlpr.shared.preprocessing.convert_examples_to_dataset(
            examples=examples,
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            phase=phase,
            verbose=True,
        )
        return [
            {
                "label_ids": datum["data_row"].label_ids,
                "label_mask": datum["data_row"].label_mask,
            }
            for datum in dataset.data
        ]

    @classmethod
    def get_labels_from_cache(cls, cache):
        return [
            {
                "label_ids": datum["data_row"].label_ids,
                "label_mask": datum["data_row"].label_mask,
            }
            for datum in cache.iter_all()
        ]


def get_preds(logits):
    return np.argmax(logits, axis=1)


def get_label_ids(task, examples):
    return np.array([task.LABEL_BIMAP.a[example.label] for example in examples])


def get_label_id_from_data_row(task, data_row):
    return task.LABEL_BIMAP.a[data_row.label_id]


def get_label_ids_from_cache(task, cache):
    return np.array([
        get_label_id_from_data_row(data_row=datum["data_row"], task=task)
        for datum in cache.iter_all()
    ])


def get_label_vals_from_cache(cache):
    return np.array([
        get_label_val_from_data_row(data_row=datum["data_row"])
        for datum in cache.iter_all()
    ])


def get_label_val_from_data_row(data_row):
    return data_row.label


def get_multiple_choice_label_ids(task, examples):
    return np.array([task.CHOICE_BIMAP.a[example.label] for example in examples])


def get_multiple_choice_label_id_from_data_row(data_row, task):
    return task.CHOICE_BIMAP.a[data_row.label_id]


def get_multiple_choice_labels_from_cache(task, cache):
    return np.array([
        get_multiple_choice_label_id_from_data_row(data_row=datum["data_row"], task=task)
        for datum in cache.iter_all()
    ])


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


def write_val_results(results, output_dir, verbose=True, do_write_preds=True):
    os.makedirs(output_dir, exist_ok=True)
    if do_write_preds:
        if len(results["logits"].shape) == 2:
            write_preds(
                logits=results["logits"],
                output_path=os.path.join(output_dir, "val_preds.csv"),
            )
        else:
            torch.save(results["logits"], os.path.join(output_dir, "val_preds.p"))
    write_metrics(
        results=results,
        output_path=os.path.join(output_dir, "val_metrics.json"),
        verbose=verbose,
    )
