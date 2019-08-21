import collections as col
import numpy as np
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, SequentialSampler

from pyutils.display import maybe_tqdm, maybe_trange

from nlpr.shared.runner import (
    convert_examples_to_dataset,
    HybridLoader,
    complex_backpropagate,
    get_sampler,
    TrainEpochState,
    TrainGlobalState,
)
from nlpr.shared.modeling import forward_batch_basic
from nlpr.shared.train_setup import TrainSchedule
import nlpr.tasks.evaluate as evaluate
from nlpr.shared.torch_utils import compute_pred_entropy_clean


@dataclass
class RunnerParameters:
    feat_spec: int
    local_rank: int
    n_gpu: int
    fp16: bool
    learning_rate: float
    eval_batch_size: int
    max_grad_norm: float


class SimpleTaskRunner:
    def __init__(self, task, model_wrapper, optimizer_scheduler, loss_criterion,
                 device, rparams: RunnerParameters, train_schedule: TrainSchedule,
                 log_writer):
        self.task = task
        self.model_wrapper = model_wrapper
        self.optimizer_scheduler = optimizer_scheduler
        self.loss_criterion = loss_criterion
        self.device = device
        self.rparams = rparams
        self.train_schedule = train_schedule
        self.log_writer = log_writer

        # Convenience
        self.model = self.model_wrapper.model

    def run_train(self, train_examples, verbose=True):

        train_dataloader = self.get_train_dataloader(train_examples)
        train_global_state = TrainGlobalState()

        for epoch_i in \
                maybe_trange(int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=verbose):
            train_global_state.epoch = epoch_i
            self.run_train_epoch(train_dataloader, train_global_state)
            results = self.run_val(val_examples=self.task.get_val_examples())
            self.log_writer.write_entry("val_metric", {
                "epoch": train_global_state.epoch,
                "metric": results["metrics"].asdict(),
            })
            self.log_writer.flush()

    def run_train_val(self, train_examples, val_examples, verbose=True):
        epoch_result_dict = col.OrderedDict()
        train_global_state = TrainGlobalState()
        for epoch_i in maybe_trange(
                int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=verbose):
            train_global_state.epoch = epoch_i
            train_dataloader = self.get_train_dataloader(train_examples)
            self.run_train_epoch(train_dataloader, train_global_state)
            epoch_result = self.run_val(val_examples)
            del epoch_result["logits"]
            epoch_result["metrics"] = epoch_result["metrics"].asdict()
            epoch_result_dict[epoch_i] = epoch_result
        return epoch_result_dict

    def run_train_epoch(self, train_dataloader, train_global_state, verbose=True):
        for _ in self.run_train_epoch_context(
                train_dataloader=train_dataloader,
                train_global_state=train_global_state,
                verbose=verbose):
            pass

    def run_train_epoch_context(self, train_dataloader, train_global_state, verbose=True):
        train_epoch_state = TrainEpochState()
        for step, (batch, batch_metadata) in enumerate(
                maybe_tqdm(train_dataloader, desc="Training", verbose=verbose)):
            self.run_train_step(
                step=step,
                batch=batch,
                train_epoch_state=train_epoch_state,
                train_global_state=train_global_state,
            )
            yield step, batch, train_epoch_state

    def run_train_step(self, step, batch, train_epoch_state, train_global_state):
        self.model.train()
        batch = batch.to(self.device)
        logits = forward_batch_basic(
            model=self.model,
            batch=batch,
            omit_label_ids=True,
        )[0]
        loss = self.loss_criterion(logits, batch.label_ids)
        loss = self.complex_backpropagate(loss)
        loss_val = loss.item()

        train_epoch_state.tr_loss += loss_val
        train_epoch_state.nb_tr_examples += len(batch)
        train_epoch_state.nb_tr_steps += 1
        if (step + 1) % self.train_schedule.gradient_accumulation_steps == 0:
            self.optimizer_scheduler.step()
            self.model.zero_grad()
            train_epoch_state.global_step += 1
            train_global_state.global_step += 1

        self.log_writer.write_entry("loss_train", {
            "epoch": train_global_state.epoch,
            "epoch_step": train_epoch_state.global_step,
            "global_step": train_global_state.global_step,
            "loss_val": loss_val,
            "pred_entropy": compute_pred_entropy_clean(logits)
        })

    def run_val(self, val_examples, verbose=True):
        if not self.rparams.local_rank == -1:
            return
        self.model.eval()
        val_dataloader = self.get_eval_dataloader(val_examples)
        total_eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits = []
        for step, (batch, batch_metadata) in enumerate(
                maybe_tqdm(val_dataloader, desc="Evaluating (Val)", verbose=verbose)):
            batch = batch.to(self.device)

            with torch.no_grad():
                logits = forward_batch_basic(
                    model=self.model,
                    batch=batch,
                    omit_label_ids=True,
                )[0]
                tmp_eval_loss = self.loss_criterion(logits, batch.label_ids)

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
            "metrics": evaluate.compute_task_metrics(self.task, all_logits, val_examples),
        }

    def run_test(self, test_examples, verbose=True):
        test_dataloader = self.get_eval_dataloader(test_examples)
        self.model.eval()
        all_logits = []
        for step, (batch, batch_metadata) in enumerate(
                maybe_tqdm(test_dataloader, desc="Predictions (Test)", verbose=verbose)):
            batch = batch.to(self.device)
            with torch.no_grad():
                logits = forward_batch_basic(
                    model=self.model,
                    batch=batch,
                    omit_label_ids=True,
                )[0]
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)

        all_logits = np.concatenate(all_logits, axis=0)
        return all_logits

    def get_train_dataloader(self, train_examples, verbose=True):
        dataset_with_metadata = convert_examples_to_dataset(
            examples=train_examples,
            feat_spec=self.rparams.feat_spec,
            tokenizer=self.model_wrapper.tokenizer,
            task=self.task,
            verbose=verbose,
        )
        train_sampler = get_sampler(
            dataset=dataset_with_metadata.dataset,
            local_rank=self.rparams.local_rank,
        )
        train_dataloader = DataLoader(
            dataset=dataset_with_metadata.dataset,
            sampler=train_sampler,
            batch_size=self.train_schedule.train_batch_size,
        )
        return HybridLoader(
            dataloader=train_dataloader,
            metadata=dataset_with_metadata.metadata,
            task=self.task,
        )

    def get_eval_dataloader(self, eval_examples):
        dataset_with_metadata = convert_examples_to_dataset(
            examples=eval_examples,
            feat_spec=self.rparams.feat_spec,
            tokenizer=self.model_wrapper.tokenizer,
            task=self.task,
        )
        eval_sampler = SequentialSampler(dataset_with_metadata.dataset)
        eval_dataloader = DataLoader(
            dataset=dataset_with_metadata.dataset,
            sampler=eval_sampler,
            batch_size=self.rparams.eval_batch_size,
        )
        return HybridLoader(
            dataloader=eval_dataloader,
            metadata=dataset_with_metadata.metadata,
            task=self.task,
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
