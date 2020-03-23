from typing import Dict
from dataclasses import dataclass

import torch

from pyutils.display import maybe_tqdm
from pyutils.datastructures import InfiniteYield

from nlpr.shared.pycore import ExtendedDataClassMixin
from nlpr.shared.runner import (
    BaseRunner,
)
from nlpr.proj.simple.runner import (
    complex_backpropagate,
    get_train_dataloader_from_cache,
    get_eval_dataloader_from_cache,
)
import nlpr.shared.pycore as pycore
from nlpr.proj.jiant.modeling.primary import JiantStyleModel
import nlpr.tasks.evaluate as evaluate
from nlpr.proj.jiant.components.task_setup import JiantTaskContainer
from nlpr.constants import PHASE
import nlpr.shared.torch_utils as torch_utils


@dataclass
class RunnerParameters(pycore.ExtendedDataClassMixin):
    local_rank: int
    n_gpu: int
    fp16: bool
    max_grad_norm: float


@dataclass
class TrainState(ExtendedDataClassMixin):
    global_steps: int
    task_steps: Dict[str, int]

    @classmethod
    def from_task_name_list(cls, task_name_list):
        return cls(
            global_steps=0,
            task_steps={
                task_name: 0
                for task_name in task_name_list
            },
        )

    def step(self, task_name):
        self.task_steps[task_name] += 1
        self.global_steps += 1


class JiantRunner(BaseRunner):
    def __init__(self,
                 jiant_task_container: JiantTaskContainer,
                 jiant_model: JiantStyleModel,
                 optimizer_scheduler,
                 device,
                 rparams: RunnerParameters,
                 log_writer):
        self.jiant_task_container = jiant_task_container
        self.jiant_model = jiant_model
        self.optimizer_scheduler = optimizer_scheduler
        self.device = device
        self.rparams = rparams
        self.log_writer = log_writer

        self.model = self.jiant_model

    def run_train_context(self, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        train_state = TrainState.from_task_name_list(list(self.jiant_task_container.task_dict))
        for _ in maybe_tqdm(range(self.jiant_task_container.global_train_config.max_steps),
                            desc="Training", verbose=verbose):
            self.run_train_step(train_dataloader_dict=train_dataloader_dict, train_state=train_state)
            yield train_state

    def resume_train_context(self, train_state, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        start_position = train_state.global_steps
        for _ in maybe_tqdm(range(start_position, self.jiant_task_container.global_train_config.max_steps),
                            desc="Training", initial=start_position,
                            total=self.jiant_task_container.global_train_config.max_steps,
                            verbose=verbose):
            self.run_train_step(train_dataloader_dict=train_dataloader_dict, train_state=train_state)
            yield train_state

    def run_train_step(self, train_dataloader_dict: dict, train_state: TrainState):
        self.jiant_model.train()
        task_name, task = self.jiant_task_container.task_sampler.pop()
        task_specific_config = self.jiant_task_container.task_specific_configs[task_name]

        loss_val = 0
        for i in range(task_specific_config.gradient_accumulation_steps):
            batch, batch_metadata = train_dataloader_dict[task_name].pop()
            batch = batch.to(self.device)
            model_output = self.jiant_model(
                batch=batch,
                task=task,
                compute_loss=True,
            )
            loss = self.complex_backpropagate(
                loss=model_output.loss,
                gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
            )
            loss_val += loss.item()

        self.optimizer_scheduler.step()
        self.optimizer_scheduler.optimizer.zero_grad()

        train_state.step(task_name=task_name)
        self.log_writer.write_entry("loss_train", {
            "task": task_name,
            "task_step": train_state.task_steps[task_name],
            "global_step": train_state.global_steps,
            "loss_val": loss_val / task_specific_config.gradient_accumulation_steps,
        })

    def run_val(self, use_subset=None, verbose=True):
        evaluate_dict = {}
        val_dataloader_dict = self.get_val_dataloader_dict(use_subset=use_subset)
        val_labels_dict = self.get_val_labels_dict(use_subset=use_subset)
        for task_name, task in self.jiant_task_container.task_dict.items():
            evaluate_dict[task_name] = run_val(
                val_dataloader=val_dataloader_dict[task_name],
                val_labels=val_labels_dict[task_name],
                jiant_model=self.jiant_model,
                task=task,
                device=self.device,
                local_rank=self.rparams.local_rank,
                verbose=verbose,
            )
        return evaluate_dict

    def get_train_dataloader_dict(self):
        # Not currently supported distributed parallel
        train_dataloader_dict = {}
        for task_name, task in self.jiant_task_container.task_dict.items():
            train_cache = self.jiant_task_container.task_cache_dict[task_name]["train"]
            train_batch_size = self.jiant_task_container.task_specific_configs[task_name].train_batch_size
            train_dataloader_dict[task_name] = InfiniteYield(get_train_dataloader_from_cache(
                train_cache=train_cache,
                task=task,
                train_batch_size=train_batch_size,
            ))
        return train_dataloader_dict

    def _get_eval_dataloader_dict(self, phase, use_subset=False):
        val_dataloader_dict = {}
        for task_name, task in self.jiant_task_container.task_dict.items():
            eval_cache = self.jiant_task_container.task_cache_dict[task_name][phase]
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            val_dataloader_dict[task_name] = get_eval_dataloader_from_cache(
                eval_cache=eval_cache,
                task=task,
                eval_batch_size=task_specific_config.eval_batch_size,
                subset_num=task_specific_config.eval_subset_num if use_subset else None
            )
        return val_dataloader_dict

    def get_val_dataloader_dict(self, use_subset=False):
        return self._get_eval_dataloader_dict(phase="val", use_subset=use_subset)

    def get_val_labels_dict(self, use_subset=False):
        val_labels_dict = {}
        for task_name, task in self.jiant_task_container.task_dict.items():
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            val_labels_cache = self.jiant_task_container.task_cache_dict[task_name]["val_labels"]
            val_labels = val_labels_cache.get_all()
            if use_subset:
                val_labels = val_labels[:task_specific_config.eval_subset_num]
            val_labels_dict[task_name] = val_labels
        return val_labels_dict

    def get_test_dataloader_dict(self):
        return self._get_eval_dataloader_dict(phase=PHASE.TEST)

    def complex_backpropagate(self, loss, gradient_accumulation_steps):
        return complex_backpropagate(
            loss=loss,
            optimizer=self.optimizer_scheduler.optimizer,
            model=self.jiant_model,
            fp16=self.rparams.fp16,
            n_gpu=self.rparams.n_gpu,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=self.rparams.max_grad_norm,
        )

    def get_runner_state(self):
        # Todo: Add fp16
        state = {
            "model": self.jiant_model.state_dict(),
            "optimizer": self.optimizer_scheduler.optimizer.state_dict(),
        }
        return state

    def load_state(self, runner_state):
        self.jiant_model.load_state_dict(runner_state["model"])
        self.optimizer_scheduler.optimizer.load_state_dict(runner_state["optimizer"])


class CheckpointSaver:
    def __init__(self, metadata, save_path):
        self.metadata = metadata
        self.save_path = save_path

    def save(self, runner_state: dict, metarunner_state: dict):
        to_save = {
            "runner_state": runner_state,
            "metarunner_state": metarunner_state,
            "metadata": self.metadata
        }
        torch_utils.safe_save(to_save, self.save_path)


def run_val(val_dataloader,
            val_labels,
            jiant_model: JiantStyleModel,
            task,
            device, local_rank, verbose):
    # Reminder:
    #   val_dataloader contains mostly PyTorch-relevant info
    #   val_labels might contain more details information needed for full evaluation
    if not local_rank == -1:
        return
    jiant_model.eval()
    total_eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    for step, (batch, batch_metadata) in enumerate(
            maybe_tqdm(val_dataloader, desc="Evaluating (Val)", verbose=verbose)):
        batch = batch.to(device)

        with torch.no_grad():
            model_output = jiant_model(
                batch=batch,
                task=task,
                compute_loss=True,
            )
        batch_logits = model_output.logits.detach().cpu().numpy()
        batch_loss = model_output.loss.mean().item()
        total_eval_loss += batch_loss
        eval_accumulator.update(
            batch_logits=batch_logits,
            batch_loss=batch_loss,
            batch=batch,
        )

        nb_eval_examples += len(batch)
        nb_eval_steps += 1
    eval_loss = total_eval_loss / nb_eval_steps
    return {
        "accumulator": eval_accumulator,
        "loss": eval_loss,
        "metrics": evaluation_scheme.compute_metrics_from_accumulator(
            task=task,
            accumulator=eval_accumulator,
            labels=val_labels,
            tokenizer=jiant_model.tokenizer,
        ),
    }
