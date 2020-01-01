import collections
import numpy as np
import os
import pandas as pd
import tqdm
import scipy.special
from dataclasses import dataclass
import typing

import torch
import torch.nn.functional as F

import nlpr.shared.distributed as distributed
import nlpr.shared.model_setup as model_setup
import nlpr.shared.train_setup as train_setup
import nlpr.shared.model_resolution as model_resolution
import nlpr.proj.simple.runner as simple_runner
import nlpr.shared.metarunner as metarunner
import nlpr.tasks as tasks
import nlpr.tasks.evaluate as evaluate
from nlpr.shared.runner import BaseRunner

import zproto.zlogv1 as zlogv1


class RunnerCreator:
    def __init__(self, task,
                 model_type, model_path, model_config_path, model_tokenizer_path, model_load_mode,
                 learning_rate, warmup_steps, warmup_proportion, optimizer_type,
                 train_batch_size, eval_batch_size, num_train_epochs, max_steps,
                 gradient_accumulation_steps, max_grad_norm, max_seq_length,
                 local_rank, fp16, fp16_opt_level, device, n_gpu,
                 verbose=False):
        self.task = task
        self.model_type = model_type
        self.model_path = model_path
        self.model_config_path = model_config_path
        self.model_tokenizer_path = model_tokenizer_path
        self.model_load_mode = model_load_mode

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.warmup_proportion = warmup_proportion
        self.optimizer_type = optimizer_type

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.max_seq_length = max_seq_length

        self.local_rank = local_rank
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level

        self.device = device
        self.n_gpu = n_gpu

        self.verbose = verbose

    def create(self, num_train_examples, log_writer=zlogv1.PRINT_LOGGER):
        train_schedule = train_setup.get_train_schedule(
            num_train_examples=num_train_examples,
            max_steps=self.max_steps,
            num_train_epochs=self.num_train_epochs,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            per_gpu_train_batch_size=self.train_batch_size,
            n_gpu=self.n_gpu,
        )
        with distributed.only_first_process(local_rank=self.local_rank):
            # load the model
            model_class_spec = model_resolution.resolve_model_setup_classes(
                model_type=self.model_type,
                task_type=self.task.TASK_TYPE,
            )
            model_wrapper = model_setup.simple_model_setup(
                model_type=self.model_type,
                model_class_spec=model_class_spec,
                config_path=self.model_config_path,
                tokenizer_path=self.model_tokenizer_path,
                task=self.task,
            )
            model_setup.simple_load_model_path(
                model=model_wrapper.model,
                model_load_mode=self.model_load_mode,
                model_path=self.model_path,
            )
            model_wrapper.model.to(self.device)
        optimizer_scheduler = model_setup.create_optimizer(
            model=model_wrapper.model,
            learning_rate=self.learning_rate,
            t_total=train_schedule.t_total,
            warmup_steps=self.warmup_steps,
            warmup_proportion=self.warmup_proportion,
            optimizer_type=self.optimizer_type,
            verbose=self.verbose,
        )
        model_setup.special_model_setup(
            model_wrapper=model_wrapper,
            optimizer_scheduler=optimizer_scheduler,
            fp16=self.fp16, fp16_opt_level=self.fp16_opt_level,
            n_gpu=self.n_gpu, local_rank=self.local_rank,
        )
        loss_criterion = train_setup.resolve_loss_function(task_type=self.task.TASK_TYPE)
        rparams = simple_runner.RunnerParameters(
            feat_spec=model_resolution.build_featurization_spec(
                model_type=self.model_type,
                max_seq_length=self.max_seq_length,
            ),
            local_rank=self.local_rank,
            n_gpu=self.n_gpu,
            fp16=self.fp16,
            learning_rate=self.learning_rate,
            eval_batch_size=self.eval_batch_size,
            max_grad_norm=self.max_grad_norm,
        )
        runner = simple_runner.SimpleTaskRunner(
            task=self.task,
            model_wrapper=model_wrapper,
            optimizer_scheduler=optimizer_scheduler,
            loss_criterion=loss_criterion,
            device=self.device,
            rparams=rparams,
            train_schedule=train_schedule,
            log_writer=log_writer,
        )
        return runner


@dataclass
class MetaRunnerParameters:
    partial_eval_number: int
    save_every_steps: int
    eval_every_steps: int
    output_dir: str
    do_save: bool


@dataclass
class NTrainingRunnerParameters:
    num_models: int
    num_iter: int
    with_disagreement: bool
    confidence_threshold: typing.Union[float, None]


@dataclass
class TrainingSet:
    task: typing.Any
    labeled_examples: list
    unlabeled_examples: list
    labeled_indices: np.ndarray
    chosen_examples: np.ndarray
    chosen_preds: np.ndarray

    def get_training_examples(self, i):
        pseudolabeled_examples = [
            example.new(
                label=self.task.LABELS[self.chosen_preds[train_idx, i]]
            )
            for train_idx, example in enumerate(self.unlabeled_examples)
            if self.chosen_examples[train_idx, i]
        ]
        return self.labeled_examples + pseudolabeled_examples

    @classmethod
    def create_initial_training_set(cls, task, labeled_examples, unlabeled_examples, num_models,
                                    do_bootstrap=False):
        base_labeled_indices = np.arange(len(labeled_examples))
        if do_bootstrap:
            labeled_indices = np.random.choice(
                base_labeled_indices, size=(len(labeled_examples), num_models),
            )
        else:
            labeled_indices = np.stack([base_labeled_indices] * num_models, axis=1)
        return cls(
            task=task,
            labeled_examples=labeled_examples,
            unlabeled_examples=unlabeled_examples,
            labeled_indices=labeled_indices,
            chosen_examples=np.zeros([len(unlabeled_examples), num_models]),
            chosen_preds=np.zeros([len(unlabeled_examples), num_models]),
        )


class NTrainingRunner(BaseRunner):
    def __init__(self,
                 runner_creator: RunnerCreator,
                 labeled_examples, unlabeled_examples,
                 rparams: NTrainingRunnerParameters,
                 meta_runner_parameters: MetaRunnerParameters,
                 log_writer: zlogv1.BaseZLogger):
        self.runner_creator = runner_creator
        self.labeled_examples = labeled_examples
        self.unlabeled_examples = unlabeled_examples
        self.rparams = rparams
        self.meta_runner_parameters = meta_runner_parameters
        self.log_writer = log_writer

        self.task = self.runner_creator.task

    def train(self):
        training_set = TrainingSet.create_initial_training_set(
            task=self.task,
            labeled_examples=self.labeled_examples,
            unlabeled_examples=self.unlabeled_examples,
            num_models=self.rparams.num_models,
            do_bootstrap=True,
        )
        self.log_writer.write_entry("misc", {
            "key": "labeled_indices",
            "data": pd.DataFrame(training_set.labeled_indices).to_dict(),
        })
        runner_ls = None
        for i in range(self.rparams.num_iter):
            print(f"\n\nITER {i}:\n\n")
            new_training_set, runner_ls = self.n_training_step(
                step=i,
                training_set=training_set,
            )
            self.log_writer.write_entry("iter_stats", {
                "iter_i": i,
                "num_chosen": pd.Series(new_training_set.chosen_examples.sum(0)).tolist(),
            }, do_print=True)
            self.log_writer.flush()
            if np.all(new_training_set.chosen_examples == training_set.chosen_examples):
                print("Full Agreement")
                break
            training_set = new_training_set
        return runner_ls

    def n_training_step(self, step, training_set: TrainingSet):
        logits_ls = []
        runner_ls = []
        for i in tqdm.trange(self.rparams.num_models):
            runner_train_examples = training_set.get_training_examples(i)
            sub_log_writer = self.get_sub_log_writer(f"runner{step}__{i}")
            with sub_log_writer.log_context():
                runner = self.runner_creator.create(
                    num_train_examples=len(runner_train_examples),
                    log_writer=sub_log_writer,
                )
                val_examples = self.task.get_val_examples()
                mrunner = metarunner.MetaRunner(
                    runner=runner,
                    train_examples=runner_train_examples,
                    # quick and dirty
                    val_examples=val_examples[:self.meta_runner_parameters.partial_eval_number],
                    should_save_func=metarunner.get_should_save_func(
                        self.meta_runner_parameters.save_every_steps),
                    should_eval_func=metarunner.get_should_eval_func(
                        self.meta_runner_parameters.eval_every_steps),
                    output_dir=self.meta_runner_parameters.output_dir,
                    verbose=True,
                    save_best_model=self.meta_runner_parameters.do_save,
                    load_best_model=True,
                    log_writer=self.log_writer,
                )
                mrunner.train_val_save_every()
                logits = runner.run_test(self.unlabeled_examples)
                runner_save_memory(runner)
            logits_ls.append(logits)
            runner_ls.append(runner)

        all_logits = np.stack(logits_ls, axis=1)
        self.log_writer.write_obj("sub_runner_logits", all_logits, {
            "step": step,
        })
        self.log_writer.flush()

        chosen_examples, chosen_preds = get_n_training_pseudolabels(
            all_logits=all_logits,
            with_disagreement=self.rparams.with_disagreement,
            confidence_threshold=self.rparams.confidence_threshold,
        )

        new_training_set = TrainingSet(
            task=training_set.task,
            labeled_examples=training_set.labeled_examples,
            unlabeled_examples=training_set.unlabeled_examples,
            labeled_indices=training_set.labeled_indices,
            chosen_examples=chosen_examples,
            chosen_preds=chosen_preds,
        )
        return new_training_set, runner_ls

    def get_sub_log_writer(self, key):
        if isinstance(self.log_writer, zlogv1.ZLogger):
            return zlogv1.ZLogger(fol_path=os.path.join(self.log_writer.fol_path, "sub", key))
        else:
            return self.log_writer


def runner_save_memory(runner):
    runner.model_wrapper.model = runner.model_wrapper.model.to(torch.device("cpu"))
    runner.optimizer_scheduler = None


def runner_reactivate(runner):
    runner.model_wrapper.model = runner.model_wrapper.model.to(runner.device)


def get_preds_from_n_logits_cube(logits_cube):
    if isinstance(logits_cube, list):
        logits_cube = np.stack(logits_cube, axis=1)
    return np.array([get_pred_from_n_logits_slice(logits_slice) for logits_slice in logits_cube])


def get_pred_from_n_logits_slice(logits_slice):
    # logits_slice: [num_models, num_classes]
    counter = collections.Counter(np.argmax(logits_slice, axis=1))
    # Get 0 entries
    for i in range(logits_slice.shape[-1]):
        counter[i] += 0
    most_common_ls = counter.most_common()
    if most_common_ls[0][1] > most_common_ls[1][1]:
        return most_common_ls[0][0]
    else:
        # Averaging probabilities
        return np.argmax(scipy.special.softmax(logits_slice, axis=1).mean(0))


def evaluate_n_logits(task, logits_cube, examples):
    if task.TASK_TYPE == tasks.TaskTypes.CLASSIFICATION:
        preds = get_preds_from_n_logits_cube(logits_cube)
        return evaluate.compute_task_metrics_from_classification_preds(
            task=task, preds=preds, examples=examples)
    elif task.TASK_TYPE == tasks.TaskTypes.REGRESSION:
        preds = logits_cube.mean(axis=1)
        return evaluate.compute_task_metrics(
            task=task, logits=preds, examples=examples)
    else:
        raise KeyError(task.TASK_TYPE)


def write_n_training_val_results(task, logits_ls, val_examples, output_dir, verbose=True):
    logits_cube = np.stack(logits_ls, axis=1)
    metrics = evaluate_n_logits(
        task=task,
        logits_cube=logits_cube,
        examples=val_examples,
    )
    torch.save(
        logits_cube,
        os.path.join(output_dir, "val_preds.p"),
    )
    evaluate.write_metrics(
        results={
            "metrics": metrics
        },
        output_path=os.path.join(output_dir, "val_metrics.json"),
        verbose=verbose,
    )


def get_n_training_pseudolabels(all_logits, with_disagreement=False, null_value=-1,
                                confidence_threshold=None):
    # all logits: [n, num_models, num_classes]
    num_examples, num_models, num_classes = all_logits.shape
    above_conf_threshold = np.ones([num_examples, num_models]).astype(bool)
    if confidence_threshold is not None:
        softmax_preds = F.softmax(torch.from_numpy(all_logits), dim=-1).numpy()
        max_softmax_preds = np.max(softmax_preds, axis=-1)
        above_conf_threshold &= (max_softmax_preds > confidence_threshold)
    all_preds = np.argmax(all_logits, axis=-1)

    all_above_conf_threshold_preds = all_preds.copy()
    all_above_conf_threshold_preds[~above_conf_threshold] = null_value

    chosen_examples_ls = []
    chosen_preds_ls = []
    for i in range(num_models):
        # Extract some preds
        others_selector = np.arange(num_models) != i
        others_preds = all_above_conf_threshold_preds[:, others_selector]
        this_preds = all_preds[:, i]
        one_other_preds = others_preds[:, 0]

        # Check that all others_preds agree with an arbitrarily chosen other column
        others_agreement = np.all((one_other_preds[:, np.newaxis] == others_preds), axis=1)
        # Check that that column's preds aren't invalid
        # (Some subtlety here. If any of the other columns are below threshold, there're 2 cases:
        #  1. All of them are below threshold, in which case others_agreement[i]=True, so
        #     we need to set it to false here. If they're all below, we just need to check
        #     the arbitrary column.
        #  2. Only some are under threshold, in which case others_agreement[i]=False, so
        #     we don't need to do anything
        if confidence_threshold is not None:
            others_agreement &= (one_other_preds != null_value)

        if with_disagreement:
            chosen_idx = others_agreement & (one_other_preds != this_preds)
        else:
            chosen_idx = others_agreement

        chosen_examples_ls.append(chosen_idx)
        chosen_preds = np.array(others_preds)[:, 0]
        chosen_preds[~chosen_idx] = null_value  # For safety
        chosen_preds_ls.append(chosen_preds)

    all_chosen_examples = np.stack(chosen_examples_ls, axis=1)
    all_chosen_preds = np.stack(chosen_preds_ls, axis=1)

    return all_chosen_examples, all_chosen_preds
