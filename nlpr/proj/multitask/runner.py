import numpy as np
from dataclasses import dataclass

import torch

from pyutils.display import maybe_tqdm

from nlpr.shared.runner import (
    BaseRunner,
    complex_backpropagate,
    TrainGlobalState,
    optim_step_grad_accum,
    get_train_dataloader,
    get_eval_dataloader,
)
from nlpr.shared.modeling.models import compute_loss_from_model_output
from nlpr.shared.train_setup import TrainSchedule
import nlpr.tasks.evaluate as evaluate
from nlpr.shared.torch_utils import compute_pred_entropy_clean
from nlpr.proj.multitask.modeling import forward_batch_delegate


@dataclass
class RunnerParameters:
    feat_spec: int
    local_rank: int
    n_gpu: int
    fp16: bool
    learning_rate: float
    eval_batch_size: int
    max_grad_norm: float


class SimpleJointDataloader:
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.length_dict = {
            task_name: len(dl)
            for task_name, dl in self.dataloader_dict.items()
        }

    def __len__(self):
        return sum(self.length_dict.values())

    def __iter__(self):
        remaining_counts = np.array(list(self.length_dict.values()))
        task_name_ls = list(self.length_dict.keys())
        iterator_dict = {task_name: iter(dl) for task_name, dl in self.dataloader_dict.items()}
        while remaining_counts.sum() > 0:
            task_i = np.random.choice(
                a=np.arange(len(task_name_ls)),
                p=remaining_counts/remaining_counts.sum(),
            )
            task_name = task_name_ls[int(task_i)]
            yield task_name, next(iterator_dict[task_name])
            remaining_counts[task_i] -= 1


class MultiTaskRunner(BaseRunner):
    def __init__(self, task_dict, model_wrapper, optimizer_scheduler, loss_criterion_dict,
                 device, rparams: RunnerParameters, train_schedule: TrainSchedule,
                 log_writer):
        self.task_dict = task_dict
        self.model_wrapper = model_wrapper
        self.optimizer_scheduler = optimizer_scheduler
        self.loss_criterion_dict = loss_criterion_dict
        self.device = device
        self.rparams = rparams
        self.train_schedule = train_schedule
        self.log_writer = log_writer

        # Convenience
        self.model = self.model_wrapper.model

    def run_train_epoch_context(self, train_dataloader,
                                train_global_state: TrainGlobalState, verbose=True):
        for task_name, (batch, batch_metadata) in maybe_tqdm(train_dataloader, desc="Training", verbose=verbose):
            self.run_train_step(
                task_name=task_name,
                batch=batch,
                train_global_state=train_global_state,
            )
            yield batch, train_global_state
        train_global_state.step_epoch()

    def run_train_step(self, task_name, batch, train_global_state):
        self.model.train()
        batch = batch.to(self.device)
        task = self.task_dict[task_name]
        logits = forward_batch_delegate(
            model=self.model,
            batch=batch,
            omit_label_ids=True,
            task_type=task.TASK_TYPE,
            task_name=task_name,
        )[0]
        loss = compute_loss_from_model_output(
            logits=logits,
            loss_criterion=self.loss_criterion_dict[task_name],
            batch=batch,
            task_type=task.TASK_TYPE,
        )

        loss = self.complex_backpropagate(loss)
        loss_val = loss.item()

        optim_step_grad_accum(
            optimizer_scheduler=self.optimizer_scheduler,
            train_global_state=train_global_state,
            gradient_accumulation_steps=self.train_schedule.gradient_accumulation_steps,
        )
        self.log_writer.write_entry("loss_train", {
            "epoch": train_global_state.epoch,
            "epoch_step": train_global_state.epoch_step,
            "global_step": train_global_state.global_step,
            "loss_val": loss_val,
            "pred_entropy": compute_pred_entropy_clean(logits)
        })

    def run_val(self, val_examples: dict, verbose=True):
        val_results_dict = {
            task_name: self.run_single_val(
                task_val_examples=task_val_examples,
                task_name=task_name,
                model=self.model,
                loss_criterion=self.loss_criterion_dict[task_name],
                verbose=verbose,
            )
            for task_name, task_val_examples in val_examples.items()
        }
        average_major = evaluate.mean([
            val_results["metrics"].major
            for val_results in val_results_dict.values()
        ])
        collated_val_results = {"logits": {}, "loss": {}, "metrics": evaluate.Metrics(major=average_major, minor={})}
        for task_name, val_results in val_results_dict.items():
            collated_val_results["logits"][task_name] = val_results["logits"]
            collated_val_results["loss"][task_name] = val_results["loss"]
            collated_val_results["metrics"].minor[task_name] = val_results["metrics"].asdict()
        return collated_val_results

    def run_single_val(self, task_val_examples, task_name, model, loss_criterion, verbose=True):
        task = self.task_dict[task_name]
        val_dataloader = self.get_single_eval_dataloader(
            eval_examples=task_val_examples,
            task=task,
        )
        if not self.rparams.local_rank == -1:
            return
        model.eval()
        total_eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits = []
        for step, (batch, batch_metadata) in enumerate(
                maybe_tqdm(val_dataloader, desc="Evaluating (Val)", verbose=verbose)):
            batch = batch.to(self.device)

            with torch.no_grad():
                logits = forward_batch_delegate(
                    model=model,
                    batch=batch,
                    omit_label_ids=True,
                    task_type=task.TASK_TYPE,
                    task_name=task_name,
                )[0]
                tmp_eval_loss = compute_loss_from_model_output(
                    logits=logits,
                    loss_criterion=loss_criterion,
                    batch=batch,
                    task_type=task.TASK_TYPE,
                )

            logits = logits.detach().cpu().numpy()
            total_eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += len(batch)
            nb_eval_steps += 1
            all_logits.append(logits)
        eval_loss = total_eval_loss / nb_eval_steps
        all_logits = np.concatenate(all_logits, axis=0)

        return {
            "logits": all_logits,
            "loss": eval_loss,
            "metrics": evaluate.compute_task_metrics(task, all_logits, task_val_examples),
        }

    def get_train_dataloader(self, train_examples, verbose=True):
        train_dataloader_dict = {
            task_name: self.get_single_train_dataloader(
                train_examples=task_train_examples,
                task=self.task_dict[task_name],
                verbose=verbose
            )
            for task_name, task_train_examples in train_examples.items()
        }
        return SimpleJointDataloader(train_dataloader_dict)

    def get_single_train_dataloader(self, train_examples, task, verbose=True):
        return get_train_dataloader(
            train_examples=train_examples,
            task=task,
            tokenizer=self.model_wrapper.tokenizer,
            feat_spec=self.rparams.feat_spec,
            local_rank=self.rparams.local_rank,
            train_batch_size=self.train_schedule.train_batch_size,
            verbose=verbose,
        )

    def get_single_eval_dataloader(self, eval_examples, task):
        return get_eval_dataloader(
            eval_examples=eval_examples,
            task=task,
            tokenizer=self.model_wrapper.tokenizer,
            feat_spec=self.rparams.feat_spec,
            eval_batch_size=self.rparams.eval_batch_size,
        )

    def complex_backpropagate(self, loss):
        return complex_backpropagate(
            loss=loss,
            optimizer=self.optimizer_scheduler.optimizer,
            model=self.model,
            fp16=self.rparams.fp16,
            n_gpu=self.rparams.n_gpu,
            gradient_accumulation_steps=self.train_schedule.gradient_accumulation_steps,
            max_grad_norm=self.rparams.max_grad_norm,
        )
