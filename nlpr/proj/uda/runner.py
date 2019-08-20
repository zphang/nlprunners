import numpy as np
from dataclasses import dataclass
from typing import Union, List

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from pyutils.display import maybe_tqdm, maybe_trange

from nlpr.shared.train_setup import TrainSchedule
from nlpr.shared.runner import (
    convert_examples_to_dataset,
    HybridLoader,
    complex_backpropagate,
    get_sampler,
    TrainEpochState,
    TrainGlobalState,
)
from nlpr.shared.modeling import forward_batch_basic
import nlpr.tasks.evaluate as evaluate
from nlpr.proj.uda import uda_ops
from nlpr.shared.torch_utils import get_val, compute_pred_entropy_clean


@dataclass
class TrainDataTriplet:
    sup: object
    unsup_orig: object
    unsup_aug: object


@dataclass
class UnsupDataLoaders:
    unsup_orig: Union[DataLoader, List[None]]
    unsup_aug: Union[DataLoader, List[None]]
    metadata: dict


@dataclass
class RunnerParameters:
    feat_spec: int
    local_rank: int
    n_gpu: int
    fp16: bool
    learning_rate: float
    eval_batch_size: int
    max_grad_norm: float


@dataclass
class UDAParameters:
    use_unsup: bool
    unsup_ratio: int
    tsa: bool
    tsa_schedule: str
    uda_softmax_temp: float
    uda_confidence_thresh: float
    uda_coeff: float


class UDARunner:
    def __init__(self, task, model_wrapper, optimizer_scheduler, loss_criterion,
                 device, rparams: RunnerParameters, uda_params: UDAParameters,
                 train_schedule: TrainSchedule, log_writer):
        self.task = task
        self.model_wrapper = model_wrapper
        self.optimizer_scheduler = optimizer_scheduler
        self.loss_criterion = loss_criterion
        self.device = device
        self.rparams = rparams
        self.uda_params = uda_params
        self.train_schedule = train_schedule
        self.log_writer = log_writer

        # Convenience
        self.model = self.model_wrapper.model

    def run_train(self, task_data, verbose=True):
        train_global_state = TrainGlobalState()
        sup_dataloader = self.get_sup_dataloader(
            task_data=task_data,
            verbose=verbose,
        )

        for epoch_i in maybe_trange(int(self.train_schedule.num_train_epochs),
                                    desc="Epoch", verbose=verbose):
            train_global_state.epoch = epoch_i
            unsup_dataloaders = self.get_unsup_dataloaders(
                sup_dataloader=sup_dataloader,
                task_data=task_data,
            )
            if self.uda_params.use_unsup:
                self.log_writer.write_entry("misc", {
                    "unsup_indices": [int(x) for x in unsup_dataloaders.metadata["unsup_indices"]],
                    "unsup_aug_set": [int(x) for x in unsup_dataloaders.metadata["unsup_aug_set"]],
                })
                self.log_writer.flush()
            dataloader_triplet = self.form_dataloader_triplet(
                sup_dataloader=sup_dataloader,
                unsup_orig_loader=unsup_dataloaders.unsup_orig,
                unsup_aug_loader=unsup_dataloaders.unsup_aug,
            )
            self.run_train_epoch(dataloader_triplet, train_global_state, verbose=verbose)
            results = self.run_val(val_examples=self.task.get_val_examples())
            self.log_writer.write_entry("val_metric", {
                "epoch": train_global_state.epoch,
                "metric": results["metrics"].asdict(),
            })

    def run_train_fixed_step(self, task_data, num_steps, verbose=True):
        assert self.train_schedule.t_total == num_steps
        assert self.train_schedule.num_train_epochs == 1

        train_global_state = TrainGlobalState()
        sup_dataloader = self.get_fixed_step_dataloader(
            examples=task_data["sup"]["train"],
            batch_size=self.train_schedule.train_batch_size,
            num_steps=num_steps,
            verbose=verbose,
        )
        unsup_dataloaders = self.get_unsup_dataloaders(
            sup_dataloader=sup_dataloader,
            task_data=task_data,
        )
        # Todo: log the log data!
        if self.uda_params.use_unsup:
            self.log_writer.write_entry("misc", {
                "unsup_indices": [int(x) for x in unsup_dataloaders.metadata["unsup_indices"]],
                "unsup_aug_set": [int(x) for x in unsup_dataloaders.metadata["unsup_aug_set"]],
            })
            self.log_writer.flush()
        dataloader_triplet = self.form_dataloader_triplet(
            sup_dataloader=sup_dataloader,
            unsup_orig_loader=unsup_dataloaders.unsup_orig,
            unsup_aug_loader=unsup_dataloaders.unsup_aug,
        )
        self.run_train_epoch(dataloader_triplet, train_global_state, verbose=verbose)

    def run_train_epoch(self, dataloader_triplet, train_global_state, verbose=True):
        for _ in self.run_train_epoch_context(dataloader_triplet, train_global_state, verbose=verbose):
            pass

    def run_train_epoch_context(self, dataloader_triplet, train_global_state, verbose=True):
        self.model.train()
        train_epoch_state = TrainEpochState()
        train_iterator = enumerate(maybe_tqdm(zip(
            dataloader_triplet.sup,
            dataloader_triplet.unsup_orig,
            dataloader_triplet.unsup_aug
        ), total=len(dataloader_triplet.sup), desc="Training", verbose=verbose))
        for step, (sup_batch, unsup_orig_batch, unsup_aug_batch) in train_iterator:
            batch_triplet = TrainDataTriplet(
                # batch, batch_metadata hack
                sup=sup_batch,
                unsup_orig=unsup_orig_batch,
                unsup_aug=unsup_aug_batch,
            )
            self.run_train_step(
                step=step,
                batch_triplet=batch_triplet,
                train_epoch_state=train_epoch_state,
                train_global_state=train_global_state,
            )
            yield step, batch_triplet, train_epoch_state

    def run_train_step(self, step, batch_triplet, train_epoch_state, train_global_state):
        example_count = len(batch_triplet.sup)
        sup_loss, sup_logits = uda_ops.sup_train_step(
            model=self.model,
            sup_batch=batch_triplet.sup[0].to(self.device),
            task=self.task,
            global_step=train_epoch_state.global_step,
            train_schedule=self.train_schedule,
            uda_params=self.uda_params,
        )
        if self.uda_params.use_unsup:
            example_count += len(batch_triplet.unsup_orig[0])
            unsup_loss, unsup_orig_logits, unsup_aug_logits = uda_ops.unsup_train_step(
                model=self.model,
                unsup_orig_batch=batch_triplet.unsup_orig[0].to(self.device),
                unsup_aug_batch=batch_triplet.unsup_aug[0].to(self.device),
                uda_params=self.uda_params,
            )
            weighted_unsup_loss = self.uda_params.uda_coeff * unsup_loss
            loss = sup_loss + weighted_unsup_loss
        else:
            unsup_orig_logits, unsup_aug_logits = None, None
            weighted_unsup_loss = 0
            loss = sup_loss
        loss = self.complex_backpropagate(loss)

        train_epoch_state.tr_loss += loss.item()
        train_epoch_state.nb_tr_examples += example_count
        train_epoch_state.nb_tr_steps += 1
        if (step + 1) % self.train_schedule.gradient_accumulation_steps == 0:
            self.optimizer_scheduler.step()
            self.model.zero_grad()
            train_epoch_state.global_step += 1
            train_global_state.global_step += 1

        log_data = {
            "epoch": train_global_state.epoch,
            "epoch_step": train_epoch_state.global_step,
            "global_step": train_global_state.global_step,
            "sup": get_val(sup_loss),
            "unsup": get_val(weighted_unsup_loss),
            "total": get_val(loss),
            "sup_pred_entropy": compute_pred_entropy_clean(sup_logits),
        }
        if self.uda_params.use_unsup:
            log_data["unsup_orig_pred_entropy"] = compute_pred_entropy_clean(unsup_orig_logits)
            log_data["unsup_aug_pred_entropy"] = compute_pred_entropy_clean(unsup_aug_logits)
        self.log_writer.write_entry("loss_train", log_data)
        self.log_writer.flush()

    def run_val(self, val_examples, verbose=True):
        if not self.rparams.local_rank == -1:
            return
        self.model.eval()
        val_dataloader = self.get_dataloader(
            examples=val_examples,
            batch_size=self.rparams.eval_batch_size,
            shuffle=False,
            verbose=True,
        )
        total_eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits = []
        for step, (batch, batch_metadata) in enumerate(maybe_tqdm(
                val_dataloader, desc="Evaluating (Val)", verbose=verbose)):
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
        test_dataloader = self.get_dataloader(
            examples=test_examples,
            batch_size=self.rparams.eval_batch_size,
            shuffle=False,
            verbose=True,
        )
        self.model.eval()
        all_logits = []
        for step, batch in enumerate(maybe_tqdm(
                test_dataloader, desc="Predictions (Test)", verbose=verbose)):
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

    def get_dataset_with_metadata(self, examples, verbose):
        return convert_examples_to_dataset(
            examples=examples,
            feat_spec=self.rparams.feat_spec,
            tokenizer=self.model_wrapper.tokenizer,
            task=self.task,
            verbose=verbose,
        )

    def get_fixed_step_dataloader(self, examples, batch_size, num_steps, verbose=False):
        # Hack, very close to get_dataloader
        dataset_with_metadata = self.get_dataset_with_metadata(
            examples=examples,
            verbose=verbose,
        )

        sampler = RandomSampler(
            data_source=dataset_with_metadata.dataset,
            replacement=True,
            num_samples=num_steps * batch_size,
        )
        dataloader = DataLoader(
            dataset=dataset_with_metadata.dataset,
            sampler=sampler,
            batch_size=batch_size,
        )
        return HybridLoader(
            dataloader=dataloader,
            metadata=dataset_with_metadata.metadata,
            task=self.task,
        )

    def get_dataloader(self, examples, batch_size, shuffle, verbose=False):
        dataset_with_metadata = self.get_dataset_with_metadata(
            examples=examples,
            verbose=verbose,
        )
        if shuffle:
            sampler = get_sampler(
                dataset=dataset_with_metadata.dataset,
                local_rank=self.rparams.local_rank,
            )
        else:
            sampler = SequentialSampler(dataset_with_metadata.dataset)

        dataloader = DataLoader(
            dataset=dataset_with_metadata.dataset,
            sampler=sampler,
            batch_size=batch_size,
        )
        return HybridLoader(
            dataloader=dataloader,
            metadata=dataset_with_metadata.metadata,
            task=self.task,
        )

    def get_sup_dataloader(self, task_data, verbose=True):
        return self.get_dataloader(
            examples=task_data["sup"]["train"],
            batch_size=self.train_schedule.train_batch_size,
            shuffle=True,   # ??????
            verbose=verbose,
        )

    def get_unsup_dataloaders(self, sup_dataloader, task_data):
        num_unsup = (
            len(sup_dataloader)
            * self.train_schedule.train_batch_size
            * self.uda_params.unsup_ratio
        )

        if self.uda_params.use_unsup:
            unsup_indices = np.random.randint(len(task_data["unsup"]["orig"]), size=num_unsup)
            aug_indices = np.random.randint(len(task_data["unsup"]["aug"]), size=num_unsup)
            unsup_orig_examples = [task_data["unsup"]["orig"][i] for i in unsup_indices]
            unsup_aug_examples = [
                task_data["unsup"]["aug"][j][i]
                for i, j in zip(unsup_indices, aug_indices)
            ]
            unsup_orig_loader = self.get_dataloader(
                examples=unsup_orig_examples,
                batch_size=self.train_schedule.train_batch_size * self.uda_params.unsup_ratio,
                shuffle=False,
                verbose=True,
            )
            unsup_aug_loader = self.get_dataloader(
                examples=unsup_aug_examples,
                batch_size=self.train_schedule.train_batch_size * self.uda_params.unsup_ratio,
                shuffle=False,
                verbose=True,
            )
        else:
            unsup_orig_loader = unsup_aug_loader = [None] * len(sup_dataloader)
            unsup_indices = aug_indices = None

        return UnsupDataLoaders(
            unsup_orig=unsup_orig_loader,
            unsup_aug=unsup_aug_loader,
            metadata={
                "unsup_indices": unsup_indices,
                "unsup_aug_set": aug_indices,
            }
        )

    @classmethod
    def form_dataloader_triplet(cls, sup_dataloader, unsup_orig_loader, unsup_aug_loader):
        assert len(sup_dataloader) == \
               len(unsup_orig_loader) == \
               len(unsup_aug_loader)
        return TrainDataTriplet(
            sup=sup_dataloader,
            unsup_orig=unsup_orig_loader,
            unsup_aug=unsup_aug_loader,
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
