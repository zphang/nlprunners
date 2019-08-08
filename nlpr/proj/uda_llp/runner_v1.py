import numpy as np
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nlpr.shared.train_setup import TrainSchedule
from nlpr.shared.runner import (
    TrainEpochState,
)
import nlpr.proj.simple.runner as simple_runner
import nlpr.proj.llp.runner as llp_runner
import nlpr.proj.uda.runner as uda_runner

from pyutils.display import maybe_trange, maybe_tqdm


@dataclass
class LLPUDAParameters:
    uda_coeff: float
    use_unsup: bool
    unsup_ratio: int


class UDALLPRunner:
    def __init__(self, task, model_wrapper, optimizer_scheduler, loss_criterion,
                 device,
                 rparams: simple_runner.RunnerParameters,
                 llp_params: llp_runner.LlpParameters,
                 llpuda_params: LLPUDAParameters,
                 train_schedule: TrainSchedule):
        self.task = task
        self.model_wrapper = model_wrapper
        self.optimizer_scheduler = optimizer_scheduler
        self.loss_criterion = loss_criterion
        self.device = device
        self.rparams = rparams
        self.llp_params = llp_params
        self.llpuda_params = llpuda_params
        self.train_schedule = train_schedule

        self.llp_state = None

        # Convenience
        self.model = self.model_wrapper.model

    # LLP
    init_llp_state = llp_runner.LLPRunner.init_llp_state
    create_empty_llp_state = llp_runner.LLPRunner.create_empty_llp_state
    # populate_llp_state = llp_runner.LLPRunner.populate_llp_state
    compute_representation_loss = llp_runner.LLPRunner.compute_representation_loss
    run_label_propagate = llp_runner.LLPRunner.propagate_labels
    convert_examples_to_dataset = llp_runner.LLPRunner.convert_examples_to_dataset
    get_sup_dataloader = llp_runner.LLPRunner.get_train_dataloader

    # UDA
    form_dataloader_triplet = uda_runner.UDARunner.form_dataloader_triplet
    get_dataloader = uda_runner.UDARunner.get_dataloader
    get_dataset_with_metadata = uda_runner.UDARunner.get_dataset_with_metadata

    # Eval
    run_val = llp_runner.LLPRunner.run_val
    run_test = llp_runner.LLPRunner.run_test
    get_eval_dataloader = llp_runner.LLPRunner.get_eval_dataloader
    complex_backpropagate = llp_runner.LLPRunner.complex_backpropagate

    def run_train(self, task_data, verbose=True):
        train_dataset_with_metadata = self.convert_examples_to_dataset(
            examples=task_data["sup"]["train"],
            verbose=verbose,
        )
        for _ in maybe_trange(int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=verbose):
            self.run_train_epoch(train_dataset_with_metadata, task_data, verbose=verbose)

    def run_train_epoch(self, train_dataset_with_metadata, task_data, verbose=True):
        for _ in self.run_train_epoch_context(train_dataset_with_metadata, task_data, verbose=verbose):
            pass

    def run_train_epoch_context(self, train_dataset_with_metadata, task_data, verbose=True):
        self.model.train()
        train_epoch_state = TrainEpochState()
        sup_dataloader = self.get_sup_dataloader(
            train_dataset_with_metadata=train_dataset_with_metadata,
            do_override_labels=True, verbose=verbose,
        )
        unsup_dataloaders = self.get_unsup_dataloaders(
            sup_dataloader=sup_dataloader,
            task_data=task_data,
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
        for step, (sup_batch_m, unsup_orig_batch_m, unsup_aug_batch_m) in train_iterator:
            batch_m_triplet = uda_runner.TrainDataTriplet(
                sup=sup_batch_m.to(self.device),
                unsup_orig=unsup_orig_batch_m.to(self.device),
                unsup_aug=unsup_aug_batch_m.to(self.device),
            )
            self.run_train_step(
                step=step,
                batch_m_triplet=batch_m_triplet,
                train_epoch_state=train_epoch_state,
            )
            yield step, batch_m_triplet, train_epoch_state

    def run_train_step(self, step, batch_m_triplet, train_epoch_state):
        llp_loss = self.compute_llp_loss(
            sup_batch_m=batch_m_triplet.sup,
        )
        uda_loss = self.compute_uda_loss(
            unsup_orig_batch_m=batch_m_triplet.unsup_orig,
            unsup_aug_batch_m=batch_m_triplet.unsup_aug,
        )
        loss = llp_loss + self.llpuda_params.uda_coeff * uda_loss

        loss = self.complex_backpropagate(loss)

        train_epoch_state.tr_loss += loss.item()
        train_epoch_state.nb_tr_examples += len(batch_m_triplet.sup)
        train_epoch_state.nb_tr_steps += 1
        if (step + 1) % self.train_schedule.gradient_accumulation_steps == 0:
            self.optimizer_scheduler.step()
            self.model.zero_grad()
            train_epoch_state.global_step += 1

        # Update memory bank
        with torch.no_grad():
            new_embedding = self.model.forward_batch(batch_m_triplet.sup.batch).embedding
        self.llp_state.big_m_tensor[batch_m_triplet.sup.metadata["example_id"]] = (
            (1 - self.llp_params.llp_mem_bank_t)
            * self.llp_state.big_m_tensor[batch_m_triplet.sup.metadata["example_id"]]
            + self.llp_params.llp_mem_bank_t * new_embedding
        )

    def compute_llp_loss(self, sup_batch_m):
        llp_loss, _ = self.compute_representation_loss(
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

    def populate_llp_state(self, train_examples, verbose=True):
        train_dataset_with_metadata = self.convert_examples_to_dataset(train_examples, verbose=True)
        train_dataloader = self.get_sup_dataloader(
            train_dataset_with_metadata=train_dataset_with_metadata,
            use_eval_batch_size=True, do_override_labels=False, verbose=verbose,
        )

        with torch.no_grad():
            for batch, metadata in maybe_tqdm(train_dataloader, desc="Initializing big_m",
                                              verbose=verbose):
                batch = batch.to(self.device)
                embedding = self.model.forward_batch(batch).embedding
                self.llp_state.big_m_tensor[metadata["example_id"]] = embedding
        self.run_label_propagate(verbose=verbose)
        self.llp_state.all_label_confidence[self.llp_params.num_labeled:] = 0

    def get_unsup_dataloaders(self, sup_dataloader, task_data):
        num_unsup = (
            len(sup_dataloader)
            * self.train_schedule.train_batch_size
            * self.llpuda_params.unsup_ratio
        )

        if self.llpuda_params.use_unsup:
            unsup_indices = np.random.randint(len(task_data["unsup"]["orig"]), size=num_unsup)
            aug_indices = np.random.randint(len(task_data["unsup"]["aug"]), size=num_unsup)
            unsup_orig_examples = [task_data["unsup"]["orig"][i] for i in unsup_indices]
            unsup_aug_examples = [
                task_data["unsup"]["aug"][j][i]
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
