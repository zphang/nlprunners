import collections as col
import numpy as np
from dataclasses import dataclass

import torch

from pyutils.display import maybe_tqdm, maybe_trange

from nlpr.shared.runner import (
    BaseRunner,
    complex_backpropagate,
    TrainGlobalState,
    optim_step_grad_accum,
    run_val,
    get_train_dataloader_from_cache,
    get_eval_dataloader_from_cache,
)
from nlpr.shared.modeling.models import forward_batch_delegate, compute_loss_from_model_output
from nlpr.shared.train_setup import TrainSchedule
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


class SimpleTaskRunner(BaseRunner):
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

    def run_train(self, train_cache, val_cache, verbose=True):
        train_dataloader = self.get_train_dataloader(train_cache)
        train_global_state = TrainGlobalState()

        for epoch_i in \
                maybe_trange(int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=verbose):
            train_global_state.epoch = epoch_i
            self.run_train_epoch(train_dataloader, train_global_state)
            results = self.run_val(val_cache=val_cache)
            self.log_writer.write_entry("val_metric", {
                "epoch": train_global_state.epoch,
                "metric": results["metrics"].asdict(),
            })
            self.log_writer.flush()

    def run_train_val(self, train_cache, val_cache, verbose=True):
        epoch_result_dict = col.OrderedDict()
        train_global_state = TrainGlobalState()
        for epoch_i in maybe_trange(
                int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=verbose):
            train_global_state.epoch = epoch_i
            train_dataloader = self.get_train_dataloader(train_cache)
            self.run_train_epoch(train_dataloader, train_global_state)
            epoch_result = self.run_val(val_cache)
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

    def run_train_epoch_context(self, train_dataloader,
                                train_global_state: TrainGlobalState, verbose=True):
        for batch, batch_metadata in maybe_tqdm(train_dataloader, desc="Training", verbose=verbose):
            self.run_train_step(
                batch=batch,
                train_global_state=train_global_state,
            )
            yield batch, train_global_state
        train_global_state.step_epoch()

    def run_train_step(self, batch, train_global_state):
        self.model.train()
        batch = batch.to(self.device)
        logits = forward_batch_delegate(
            model=self.model,
            batch=batch,
            omit_label_id=True,
            task_type=self.task.TASK_TYPE,
        )
        loss = compute_loss_from_model_output(
            logits=logits,
            loss_criterion=self.loss_criterion,
            batch=batch,
            task_type=self.task.TASK_TYPE,
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
            # TODO: Why is this here?
            # "pred_entropy": compute_pred_entropy_clean(logits)
        })

    def run_val(self, val_cache, subset=None, verbose=True):
        return run_val(
            val_dataloader=self.get_eval_dataloader(val_cache, subset=subset),
            model=self.model,
            task=self.task,
            loss_criterion=self.loss_criterion,
            device=self.device,
            local_rank=self.rparams.local_rank,
            verbose=verbose,
        )

    def run_test(self, test_cache, verbose=True):
        test_dataloader = self.get_eval_dataloader(test_cache)
        self.model.eval()
        all_logits = []
        for step, (batch, batch_metadata) in enumerate(
                maybe_tqdm(test_dataloader, desc="Predictions (Test)", verbose=verbose)):
            batch = batch.to(self.device)
            with torch.no_grad():
                logits = forward_batch_delegate(
                    model=self.model,
                    batch=batch,
                    omit_label_id=True,
                    task_type=self.task.TASK_TYPE,
                )
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)

        all_logits = np.concatenate(all_logits, axis=0)
        return all_logits

    def get_train_dataloader(self, train_cache):
        # Not currently supported distributed parallel
        return get_train_dataloader_from_cache(
            train_cache=train_cache,
            task=self.task,
            train_batch_size=self.train_schedule.train_batch_size,
        )

    def get_eval_dataloader(self, eval_cache, subset=None):
        return get_eval_dataloader_from_cache(
            eval_cache=eval_cache,
            task=self.task,
            subset=subset,
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
