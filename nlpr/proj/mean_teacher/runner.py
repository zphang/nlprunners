import collections as col
import copy
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from pyutils.display import maybe_tqdm, maybe_trange

from nlpr.shared.runner import (
    BaseRunner,
    convert_examples_to_dataset,
    HybridLoader,
    complex_backpropagate,
    get_sampler,
    TrainGlobalState,
    optim_step_grad_accum,
)
import nlpr.shared.model_setup as model_setup
from nlpr.shared.modeling import forward_batch_delegate, compute_loss_from_model_output
from nlpr.shared.train_setup import TrainSchedule
import nlpr.tasks.evaluate as evaluate
from nlpr.shared.torch_utils import compute_pred_entropy_clean
import nlpr.proj.simple.runner as simple_runner


@dataclass
class MeanTeacherParameters:
    alpha: float
    consistency_type: str
    consistency_weight: float
    consistency_ramp_up_steps: int


def create_teacher(model_wrapper: model_setup.ModelWrapper) -> model_setup.ModelWrapper:
    return model_setup.ModelWrapper(
        model=copy.deepcopy(model_wrapper.model),
        tokenizer=model_wrapper.tokenizer,
    )


def update_ema_variables(model, ema_model, alpha, global_step):
    # From https://github.com/CuriousAI/mean-teacher/
    #       blob/bd4313d5691f3ce4c30635e50fa207f49edf16fe/pytorch/main.py
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def update_teacher(student_wrapper, teacher_wrapper, alpha, global_step):
    update_ema_variables(
        model=student_wrapper.model,
        ema_model=teacher_wrapper.model,
        alpha=alpha,
        global_step=global_step,
    )


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(global_step, mt_params: MeanTeacherParameters):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    # Maybe do epochs?
    return (
        mt_params.consistency_weight
        * sigmoid_rampup(global_step, mt_params.consistency_ramp_up_steps)
    )


def softmax_mse_loss(input_logits, target_logits):
    # From https://github.com/CuriousAI/mean-teacher/
    #       blob/bd4313d5691f3ce4c30635e50fa207f49edf16fe/pytorch/mean_teacher/losses.py
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def softmax_kl_loss(input_logits, target_logits):
    # From https://github.com/CuriousAI/mean-teacher/
    #       blob/bd4313d5691f3ce4c30635e50fa207f49edf16fe/pytorch/mean_teacher/losses.py
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def compute_raw_consistency_loss(student_logits, teacher_logits, mt_params: MeanTeacherParameters):
    if mt_params.consistency_type == "kl":
        raw_consistency_loss = softmax_kl_loss(
            input_logits=student_logits,
            target_logits=teacher_logits,
        )
    elif mt_params.consistency_type == "mse":
        raw_consistency_loss = softmax_mse_loss(
            input_logits=student_logits,
            target_logits=teacher_logits,
        )
    else:
        raise KeyError(mt_params.consistency_type)
    return raw_consistency_loss


class SimpleTaskRunner(BaseRunner):
    def __init__(self, task, model_wrapper, optimizer_scheduler, teacher_model_wrapper, loss_criterion,
                 device, rparams: simple_runner.RunnerParameters, mt_params: MeanTeacherParameters,
                 train_schedule: TrainSchedule,
                 log_writer):
        self.task = task
        self.model_wrapper = model_wrapper
        self.optimizer_scheduler = optimizer_scheduler
        self.teacher_model_wrapper = teacher_model_wrapper
        self.loss_criterion = loss_criterion
        self.device = device
        self.rparams = rparams
        self.mt_params = mt_params
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
        self.teacher_model_wrapper.model.train()
        batch = batch.to(self.device)

        # Classification
        logits = forward_batch_delegate(
            model=self.model,
            batch=batch,
            omit_label_ids=True,
            task_type=self.task.TASK_TYPE,
        )[0]
        classification_loss = compute_loss_from_model_output(
            logits=logits,
            loss_criterion=self.loss_criterion,
            batch=batch,
            task_type=self.task.TASK_TYPE,
        )

        # Consistency
        with torch.no_grad():
            teacher_logits = forward_batch_delegate(
                model=self.teacher_model_wrapper.model,
                batch=batch,
                omit_label_ids=True,
                task_type=self.task.TASK_TYPE,
            )[0]
        raw_consistency_loss = compute_raw_consistency_loss(
            student_logits=logits,
            teacher_logits=teacher_logits,
            mt_params=self.mt_params,
        )
        consistency_weight = get_current_consistency_weight(
            global_step=train_global_state.global_step,
            mt_params=self.mt_params,
        )
        consistency_loss = consistency_weight * raw_consistency_loss

        # Combine
        loss = classification_loss + consistency_loss
        loss = self.complex_backpropagate(loss)

        optim_step_grad_accum(
            optimizer_scheduler=self.optimizer_scheduler,
            train_global_state=train_global_state,
            gradient_accumulation_steps=self.train_schedule.gradient_accumulation_steps,
        )
        update_teacher(
            student_wrapper=self.model_wrapper,
            teacher_wrapper=self.teacher_model_wrapper,
            alpha=self.mt_params.alpha,
            global_step=train_global_state.global_step,
        )
        self.log_writer.write_entry("loss_train", {
            "epoch": train_global_state.epoch,
            "epoch_step": train_global_state.epoch_step,
            "global_step": train_global_state.global_step,
            "classification_loss": classification_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "total_loss": loss.item(),
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
                logits = forward_batch_delegate(
                    model=self.model,
                    batch=batch,
                    omit_label_ids=True,
                    task_type=self.task.TASK_TYPE,
                )[0]
                tmp_eval_loss = compute_loss_from_model_output(
                    logits=logits,
                    loss_criterion=self.loss_criterion,
                    batch=batch,
                    task_type=self.task.TASK_TYPE,
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
                logits = forward_batch_delegate(
                    model=self.model,
                    batch=batch,
                    omit_label_ids=True,
                    task_type=self.task.TASK_TYPE,
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
