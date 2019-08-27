import numpy as np
import os
import pandas as pd
import torch
import tqdm
from dataclasses import dataclass
import typing

import nlpr.shared.distributed as distributed
import nlpr.shared.model_setup as model_setup
import nlpr.shared.train_setup as train_setup
import nlpr.shared.model_resolution as model_resolution
import nlpr.proj.simple.runner as simple_runner

import zproto.zlogv1  as zlogv1


class RunnerCreator:
    def __init__(self, task,
                 model_type, model_path, model_config_path, model_tokenizer_path, model_load_mode,
                 learning_rate, warmup_steps, warmup_proportion,
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
            model_setup.simple_load_model(
                model=model_wrapper.model,
                model_load_mode=self.model_load_mode,
                state_dict=torch.load(self.model_path),
                verbose=self.verbose,
            )
            model_wrapper.model.to(self.device)
        optimizer_scheduler = model_setup.create_optimizer(
            model=model_wrapper.model,
            learning_rate=self.learning_rate,
            t_total=train_schedule.t_total,
            warmup_steps=self.warmup_steps,
            warmup_proportion=self.warmup_proportion,
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
class NTrainingRunnerParameters:
    num_models: int
    num_iter: int
    with_disagreement: bool


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


class NTrainingRunner:
    def __init__(self,
                 runner_creator: RunnerCreator,
                 labeled_examples, unlabeled_examples,
                 rparams: NTrainingRunnerParameters,
                 log_writer: zlogv1.BaseZLogger):
        self.runner_creator = runner_creator
        self.labeled_examples = labeled_examples
        self.unlabeled_examples = unlabeled_examples
        self.rparams = rparams
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
            })
            self.log_writer.flush()
            if np.all(new_training_set.chosen_examples == training_set.chosen_examples):
                print("Full Agreement")
                break
            training_set = new_training_set
        return runner_ls

    def n_training_step(self, step, training_set: TrainingSet):
        preds_ls = []
        runner_ls = []
        for i in tqdm.trange(self.rparams.num_models):
            runner_train_examples = training_set.get_training_examples(i)
            sub_log_writer = self.get_sub_log_writer(f"runner{step}__{i}")
            with sub_log_writer.log_context():
                runner = self.runner_creator.create(
                    num_train_examples=len(runner_train_examples),
                    log_writer=sub_log_writer,
                )
                runner.run_train(runner_train_examples, verbose=True)
                logits = runner.run_test(self.unlabeled_examples)
            preds_ls.append(np.argmax(logits, axis=-1))
            runner_ls.append(runner)

        all_preds = np.stack(preds_ls, axis=1)
        self.log_writer.write_entry("sub_runner_preds", {
            "step": step,
            "preds": pd.DataFrame(all_preds).to_dict(),
        })
        self.log_writer.flush()

        chosen_examples_ls, chosen_preds_ls = get_n_training_pseudolabels(
            all_preds=all_preds,
            with_disagreement=self.rparams.with_disagreement,
        )

        new_training_set = TrainingSet(
            task=training_set.task,
            labeled_examples=training_set.labeled_examples,
            unlabeled_examples=training_set.unlabeled_examples,
            labeled_indices=training_set.labeled_indices,
            chosen_examples=np.stack(chosen_examples_ls, axis=1),
            chosen_preds=np.stack(chosen_preds_ls, axis=1),
        )
        return new_training_set, runner_ls

    def get_sub_log_writer(self, key):
        if isinstance(self.log_writer, zlogv1.ZLogger):
            return zlogv1.ZLogger(fol_path=os.path.join(self.log_writer.fol_path, "sub", key))
        else:
            return self.log_writer


def get_n_training_pseudolabels(all_preds, with_disagreement=False, null_value=-1):
    num_models = all_preds.shape[-1]
    chosen_examples_ls = []
    chosen_preds_ls = []
    for i in range(num_models):
        others_selector = np.arange(num_models) != i
        others_preds = all_preds[:, others_selector]
        others_agreement = (others_preds[:, 0][:, np.newaxis] == others_preds)

        if with_disagreement:
            chosen_idx = np.all(
                others_agreement
                & (others_preds[:, 0] != all_preds[:, i]),
                axis=1
            )
        else:
            chosen_idx = np.all(others_agreement, axis=1)

        chosen_examples_ls.append(chosen_idx)
        chosen_preds = np.array(others_preds)[:, 0]
        chosen_preds[~chosen_idx] = null_value  # For safety
        chosen_preds_ls.append(chosen_preds)
    return chosen_examples_ls, chosen_preds_ls

"""
def run_val_write(runner_ls, val_examples):
    votes_ls = []
    loss_ls = []
    for sub_runner in runner_ls:
        val_results = sub_runner.run_val(val_examples)
        votes = np.argmax(val_results["logits"], axis=1)
        loss_ls.append(float(val_examples["loss"]))

    return {
        "loss": loss_ls,

    }
"""
