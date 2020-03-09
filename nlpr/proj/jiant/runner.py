from typing import Union

import torch

from pyutils.display import maybe_tqdm, maybe_trange

from nlpr.shared.runner import (
    BaseRunner,
)
from nlpr.proj.simple.runner import (
    RunnerParameters,
    TrainGlobalState,
    complex_backpropagate,
    optim_step_grad_accum,
    get_train_dataloader_from_cache,
    get_eval_dataloader_from_cache,
)
from nlpr.shared.train_setup import TrainSchedule
from nlpr.shared.jiant_style_model.primary import JiantStyleModel
import nlpr.tasks.evaluate as evaluate


class JiantSingleTaskRunner(BaseRunner):
    def __init__(self, task, jiant_model: JiantStyleModel, optimizer_scheduler, loss_criterion,
                 device, rparams: RunnerParameters, train_schedule: Union[TrainSchedule, None],
                 log_writer):
        self.task = task
        self.jiant_model = jiant_model
        self.optimizer_scheduler = optimizer_scheduler
        self.loss_criterion = loss_criterion
        self.device = device
        self.rparams = rparams
        self.train_schedule = train_schedule
        self.log_writer = log_writer

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
        self.jiant_model.train()
        batch = batch.to(self.device)
        logits, loss = self.jiant_model(
            batch=batch,
            task=self.task,
            compute_loss=True,
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
        })

    def run_val(self, val_cache, val_labels_cache, subset_num=None, verbose=True):
        return run_val(
            val_dataloader=self.get_eval_dataloader(
                eval_cache=val_cache,
                subset_num=subset_num
            ),
            val_labels=val_labels_cache.get_all()[:subset_num],
            jiant_model=self.jiant_model,
            task=self.task,
            device=self.device,
            local_rank=self.rparams.local_rank,
            verbose=verbose,
        )

    def get_train_dataloader(self, train_cache):
        # Not currently supported distributed parallel
        return get_train_dataloader_from_cache(
            train_cache=train_cache,
            task=self.task,
            train_batch_size=self.train_schedule.train_batch_size,
        )

    def get_eval_dataloader(self, eval_cache, subset_num=None, explicit_subset=None):
        return get_eval_dataloader_from_cache(
            eval_cache=eval_cache,
            task=self.task,
            subset_num=subset_num,
            explicit_subset=explicit_subset,
            eval_batch_size=self.rparams.eval_batch_size,
        )

    def complex_backpropagate(self, loss):
        return complex_backpropagate(
            loss=loss,
            optimizer=self.optimizer_scheduler.optimizer,
            model=self.jiant_model,
            fp16=self.rparams.fp16,
            n_gpu=self.rparams.n_gpu,
            gradient_accumulation_steps=self.train_schedule.gradient_accumulation_steps,
            max_grad_norm=self.rparams.max_grad_norm,
        )


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
            batch_logits, batch_loss = jiant_model(
                batch=batch,
                task=task,
                compute_loss=True,
            )
        batch_logits = batch_logits.detach().cpu().numpy()
        batch_loss = batch_loss.mean().item()
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
