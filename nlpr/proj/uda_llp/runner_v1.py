import numpy as np
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nlpr.shared.train_setup import TrainSchedule
from nlpr.shared.runner import (
    BaseRunner,
    TrainGlobalState,
    optim_step_grad_accum,
)
import nlpr.proj.simple.runner as simple_runner
import nlpr.proj.llp.runner as llp_runner
import nlpr.proj.uda.runner as uda_runner

from pyutils.display import maybe_trange, maybe_tqdm
from pyutils.datastructures import combine_dicts


@dataclass
class LLPUDAParameters:
    uda_coeff: float
    use_unsup: bool
    unsup_ratio: int


class UDALLPRunner(BaseRunner):
    def __init__(self, task, model_wrapper, optimizer_scheduler, loss_criterion,
                 device,
                 rparams: simple_runner.RunnerParameters,
                 llp_params: llp_runner.LlpParameters,
                 llpuda_params: LLPUDAParameters,
                 train_schedule: TrainSchedule, log_writer):
        self.task = task
        self.model_wrapper = model_wrapper
        self.optimizer_scheduler = optimizer_scheduler
        self.loss_criterion = loss_criterion
        self.device = device
        self.rparams = rparams
        self.llp_params = llp_params
        self.llpuda_params = llpuda_params
        self.train_schedule = train_schedule
        self.log_writer = log_writer

        self.llp_state = None

        # Convenience
        self.model = self.model_wrapper.model

    # LLP
    create_empty_llp_state = llp_runner.LLPRunner.create_empty_llp_state
    populate_llp_state = llp_runner.LLPRunner.populate_llp_state
    compute_representation_loss = llp_runner.LLPRunner.compute_representation_loss
    propagate_labels = llp_runner.LLPRunner.propagate_labels
    convert_examples_to_dataset = llp_runner.LLPRunner.convert_examples_to_dataset
    get_sup_dataloader = llp_runner.LLPRunner.get_train_dataloader
    zero_out_unlabeled_confidence = llp_runner.LLPRunner.zero_out_unlabeled_confidence

    # UDA
    form_dataloader_triplet = uda_runner.UDARunner.form_dataloader_triplet
    get_dataloader = uda_runner.UDARunner.get_dataloader
    get_dataset_with_metadata = uda_runner.UDARunner.get_dataset_with_metadata

    # Eval
    run_val = llp_runner.LLPRunner.run_val
    run_test = llp_runner.LLPRunner.run_test
    get_eval_dataloader = llp_runner.LLPRunner.get_eval_dataloader
    complex_backpropagate = llp_runner.LLPRunner.complex_backpropagate

    def run_train(self, train_examples, uda_task_data, verbose=True):
        train_dataset_with_metadata = self.convert_examples_to_dataset(
            examples=train_examples,
            verbose=verbose,
        )
        train_global_state = TrainGlobalState()
        for _ in maybe_trange(int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=verbose):
            self.run_train_epoch(
                train_dataset_with_metadata=train_dataset_with_metadata,
                uda_task_data=uda_task_data,
                train_global_state=train_global_state,
                verbose=verbose,
            )

    def run_train_epoch(self, train_dataset_with_metadata, uda_task_data,
                        train_global_state: TrainGlobalState,
                        verbose=True):
        for _ in self.run_train_epoch_context(
                train_dataset_with_metadata=train_dataset_with_metadata,
                uda_task_data=uda_task_data,
                train_global_state=train_global_state,
                verbose=verbose):
            pass

    def run_train_epoch_context(self, train_dataset_with_metadata, uda_task_data,
                                train_global_state: TrainGlobalState,
                                populate_after=True, verbose=True):
        self.model.train()
        sup_dataloader = self.get_sup_dataloader(
            train_dataset_with_metadata=train_dataset_with_metadata,
            do_override_labels=True, verbose=verbose,
        )
        unsup_dataloaders = self.get_unsup_dataloaders(
            sup_dataloader=sup_dataloader,
            uda_task_data=uda_task_data,
        )
        dataloader_triplet = self.form_dataloader_triplet(
            sup_dataloader=sup_dataloader,
            unsup_orig_loader=unsup_dataloaders.unsup_orig,
            unsup_aug_loader=unsup_dataloaders.unsup_aug,
        )
        train_iterator = enumerate(maybe_tqdm(zip(
            dataloader_triplet.sup,
            dataloader_triplet.unsup_orig,
            dataloader_triplet.unsup_aug
        ), total=len(dataloader_triplet.sup), desc="Training", verbose=verbose))
        for sup_batch_m, unsup_orig_batch_m, unsup_aug_batch_m in train_iterator:
            batch_m_triplet = uda_runner.TrainDataTriplet(
                sup=sup_batch_m.to(self.device),
                unsup_orig=unsup_orig_batch_m.to(self.device),
                unsup_aug=unsup_aug_batch_m.to(self.device),
            )
            self.run_train_step(
                batch_m_triplet=batch_m_triplet,
                train_global_state=train_global_state,
            )
            yield batch_m_triplet, train_global_state
        if populate_after:
            self.populate_llp_state(
                train_dataloader=sup_dataloader,
                verbose=verbose,
            )
            self.log_writer.write_entry("populate_logs", combine_dicts([
                llp_runner.populate_logs(llp_state=self.llp_state, llp_params=self.llp_params),
                {
                    "epoch": train_global_state.epoch,
                },
            ]))

    def run_train_step(self, batch_m_triplet, train_global_state: TrainGlobalState):
        llp_loss = self.compute_llp_loss(
            sup_batch_m=batch_m_triplet.sup,
        )
        uda_loss = self.compute_uda_loss(
            unsup_orig_batch_m=batch_m_triplet.unsup_orig,
            unsup_aug_batch_m=batch_m_triplet.unsup_aug,
        )
        loss = llp_loss + self.llpuda_params.uda_coeff * uda_loss
        loss = self.complex_backpropagate(loss)
        loss_val = loss.item()

        optim_step_grad_accum(
            optimizer_scheduler=self.optimizer_scheduler,
            train_global_state=train_global_state,
            gradient_accumulation_steps=self.train_schedule.gradient_accumulation_steps,
        )

        # Update memory bank
        with torch.no_grad():
            new_embedding = self.model.forward_batch(batch_m_triplet.sup.batch).embedding
        self.llp_state.big_m_tensor[batch_m_triplet.sup.metadata["example_id"]] = (
            (1 - self.llp_params.llp_mem_bank_t)
            * self.llp_state.big_m_tensor[batch_m_triplet.sup.metadata["example_id"]]
            + self.llp_params.llp_mem_bank_t * new_embedding
        )

        self.log_writer.write_entry("loss_train", {
            "epoch": train_global_state.epoch,
            "epoch_step": train_global_state.epoch_step,
            "global_step": train_global_state.global_step,
            "loss_val": loss_val,
        })

    def compute_llp_loss(self, sup_batch_m):
        llp_loss, _, _ = self.compute_representation_loss(
            batch=sup_batch_m.batch,
            batch_metadata=sup_batch_m.metadata,
        )
        return llp_loss

    def compute_uda_loss(self, unsup_orig_batch_m, unsup_aug_batch_m):
        orig_output = self.model.forward_batch(
            unsup_orig_batch_m.batch, normalize_embedding=True)
        aug_output = self.model.forward_batch(
            unsup_aug_batch_m.batch, normalize_embedding=True)
        return (1-F.cosine_similarity(
            orig_output.embedding,
            aug_output.embedding,
            dim=-1,
        )).mean()

    def get_unsup_dataloaders(self, sup_dataloader, uda_task_data):
        num_unsup = (
            len(sup_dataloader)
            * self.train_schedule.train_batch_size
            * self.llpuda_params.unsup_ratio
        )

        if self.llpuda_params.use_unsup:
            unsup_indices = np.random.randint(len(uda_task_data["unsup"]["orig"]), size=num_unsup)
            aug_indices = np.random.randint(len(uda_task_data["unsup"]["aug"]), size=num_unsup)
            unsup_orig_examples = [uda_task_data["unsup"]["orig"][i] for i in unsup_indices]
            unsup_aug_examples = [
                uda_task_data["unsup"]["aug"][j][i]
                for i, j in zip(unsup_indices, aug_indices)
            ]
            unsup_orig_loader = self.get_dataloader(
                examples=unsup_orig_examples,
                batch_size=self.train_schedule.train_batch_size * self.llpuda_params.unsup_ratio,
                shuffle=False,
                verbose=True,
            )
            unsup_aug_loader = self.get_dataloader(
                examples=unsup_aug_examples,
                batch_size=self.train_schedule.train_batch_size * self.llpuda_params.unsup_ratio,
                shuffle=False,
                verbose=True,
            )
        else:
            unsup_orig_loader = unsup_aug_loader = [None] * len(sup_dataloader)
            unsup_indices = aug_indices = None

        return uda_runner.UnsupDataLoaders(
            unsup_orig=unsup_orig_loader,
            unsup_aug=unsup_aug_loader,
            metadata={
                "unsup_indices": unsup_indices,
                "unsup_aug_set": aug_indices,
            }
        )

    def init_llp_state(self, train_examples, verbose=True, zero_out_unlabeled_confidence=True):
        self.llp_state = self.create_empty_llp_state(train_examples=train_examples)
        train_dataset_with_metadata = self.convert_examples_to_dataset(train_examples, verbose=True)
        train_dataloader = self.get_sup_dataloader(
            train_dataset_with_metadata=train_dataset_with_metadata,
            use_eval_batch_size=True, do_override_labels=False, verbose=verbose,
        )
        self.populate_llp_state(
            train_dataloader=train_dataloader,
            verbose=verbose
        )
        if zero_out_unlabeled_confidence:
            self.zero_out_unlabeled_confidence()
        self.log_writer.write_entry("populate_logs", combine_dicts([
            llp_runner.populate_logs(llp_state=self.llp_state, llp_params=self.llp_params),
            {
                "epoch": -1,
            },
        ]))
        self.log_writer.flush()
