import numpy as np

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from pyutils.display import maybe_tqdm, maybe_trange

import nlpr.proj.llp.propagate as llp_propagate
import nlpr.proj.llp.representation as llp_representation
from nlpr.shared.runner import (
    convert_examples_to_dataset,
    HybridLoader,
    complex_backpropagate,
    get_sampler,
    TrainEpochState,
)
from nlpr.shared.train_setup import TrainSchedule
import nlpr.tasks.evaluate as evaluate
import nlpr.shared.torch_utils as torch_utils


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
class LlpParameters:
    num_labeled: int

    # Hyperparams
    llp_embedding_dim: int
    llp_const_k: int
    llp_const_t: int
    llp_const_tau: float
    llp_prop_chunk_size: int
    llp_mem_bank_t: float
    llp_rep_global_agg_loss_lambda: float
    llp_embedding_norm_loss: float
    llp_compute_global_agg_loss_mode: str


@dataclass
class LlpState:
    big_m_tensor: torch.Tensor
    all_labels_tensor: torch.Tensor
    all_label_confidence: torch.Tensor


class LLPRunner:
    def __init__(self, task, model_wrapper, optimizer_scheduler, loss_criterion,
                 device, rparams: RunnerParameters, llp_params: LlpParameters,
                 train_schedule: TrainSchedule):
        self.task = task
        self.model_wrapper = model_wrapper
        self.optimizer_scheduler = optimizer_scheduler
        self.loss_criterion = loss_criterion
        self.device = device
        self.rparams = rparams
        self.llp_params = llp_params
        self.train_schedule = train_schedule

        self.llp_state = None

        # Convenience
        self.model = self.model_wrapper.model

    def init_llp_state(self, train_examples, verbose=True):
        self.llp_state = self.create_empty_llp_state(train_examples=train_examples)
        self.populate_llp_state(train_examples=train_examples, verbose=verbose)

    def create_empty_llp_state(self, train_examples):
        big_m_tensor = torch.empty([
            len(train_examples), self.llp_params.llp_embedding_dim
        ]).to(self.device)
        all_labels = evaluate.get_label_ids(
            examples=train_examples,
            task=self.task,
        )
        all_labels_tensor = torch.from_numpy(all_labels).to(self.device)
        all_label_confidence = torch.ones(len(all_labels)).to(self.device)
        return LlpState(
            big_m_tensor=big_m_tensor,
            all_labels_tensor=all_labels_tensor,
            all_label_confidence=all_label_confidence,
        )

    def populate_llp_state(self, train_examples, verbose=True):
        train_dataloader = self.get_train_dataloader(train_examples, use_eval_batch_size=True,
                                                     do_override_labels=False)
        with torch.no_grad():
            for batch, metadata in maybe_tqdm(train_dataloader, desc="Initializing big_m",
                                              verbose=verbose):
                batch = batch.to(self.device)
                embedding = self.model.forward_batch(batch).embedding
                self.llp_state.big_m_tensor[metadata["example_id"]] = embedding
        self.run_label_propagate()
        self.llp_state.all_label_confidence[self.llp_params.num_labeled:] = 0

    def run_train(self, train_examples, verbose=True):
        train_dataloader = self.get_train_dataloader(train_examples)
        for _ in maybe_trange(int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=verbose):
            self.run_train_epoch(train_dataloader, verbose=verbose)

    def run_train_val(self, train_examples, val_examples, verbose=True):
        train_dataloader = self.get_train_dataloader(train_examples)
        epoch_result_dict = OrderedDict()
        for i in maybe_trange(int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=verbose):
            self.run_train_epoch(train_dataloader, verbose=verbose)
            epoch_result = self.run_val(val_examples)
            del epoch_result["logits"]
            epoch_result["metrics"] = epoch_result["metrics"].asdict()
            epoch_result_dict[i] = epoch_result
        return epoch_result_dict

    def run_train_epoch(self, train_dataloader, verbose=True):
        for _ in self.run_train_epoch_context(train_dataloader, verbose=verbose):
            pass

    def run_train_epoch_context(self, train_dataloader, verbose=True):
        self.model.train()
        train_epoch_state = TrainEpochState()
        for step, (batch, batch_metadata) in enumerate(
                maybe_tqdm(train_dataloader, desc="Training", verbose=verbose)):
            self.run_train_step(
                step=step,
                batch=batch,
                batch_metadata=batch_metadata,
                train_epoch_state=train_epoch_state,
            )
            yield step, batch, train_epoch_state
        self.run_label_propagate()

    def run_train_step(self, step, batch, batch_metadata, train_epoch_state):
        batch = batch.to(self.device)
        loss, loss_details = self.compute_representation_loss(batch, batch_metadata)
        loss = self.complex_backpropagate(loss)

        train_epoch_state.tr_loss += loss.item()
        train_epoch_state.nb_tr_examples += len(batch)
        train_epoch_state.nb_tr_steps += 1
        if (step + 1) % self.train_schedule.gradient_accumulation_steps == 0:
            self.optimizer_scheduler.step()
            self.model.zero_grad()
            train_epoch_state.global_step += 1

        # Update memory bank
        with torch.no_grad():
            new_embedding = self.model.forward_batch(batch).embedding
        self.llp_state.big_m_tensor[batch_metadata["example_id"]] = (
                (1 - self.llp_params.llp_mem_bank_t)
                * self.llp_state.big_m_tensor[batch_metadata["example_id"]]
                + self.llp_params.llp_mem_bank_t * new_embedding
        )
        return loss_details

    def compute_representation_loss(self, batch, batch_metadata):
        output = self.model.forward_batch(batch, normalize_embedding=False)

        weight = self.llp_state.all_label_confidence[batch_metadata["example_id"]]
        per_example_pred_loss = F.cross_entropy(output.logits, batch.label_ids, reduction="none")
        if self.llp_params.llp_compute_global_agg_loss_mode == "v1":
            per_example_global_agg_loss = llp_representation.compute_global_agg_loss(
                embedding=torch_utils.normalize_embedding_tensor(output.embedding),
                label_ids=batch.label_ids,
                big_m_tensor=self.llp_state.big_m_tensor,
                all_labels_tensor=self.llp_state.all_labels_tensor,
                const_tau=self.llp_params.llp_const_tau,
            )
        elif self.llp_params.llp_compute_global_agg_loss_mode == "v2":
            per_example_global_agg_loss = llp_representation.compute_global_agg_loss_v2(
                embedding=torch_utils.normalize_embedding_tensor(output.embedding),
                label_ids=batch.label_ids,
                big_m_tensor=self.llp_state.big_m_tensor,
                all_labels_tensor=self.llp_state.all_labels_tensor,
                const_tau=self.llp_params.llp_const_tau,
                batch_indices=torch.tensor(batch_metadata["example_id"]).to(self.device),
            )
        else:
            raise KeyError(self.llp_params.llp_compute_global_agg_loss_mode)
        per_example_embedding_norm_loss = torch_utils.embedding_norm_loss(output.embedding)

        per_example_representation_loss = (
            per_example_pred_loss
            + self.llp_params.llp_rep_global_agg_loss_lambda * per_example_global_agg_loss
            + self.llp_params.llp_embedding_norm_loss * per_example_embedding_norm_loss
        )
        representation_loss = (per_example_representation_loss * weight).mean()
        loss_details = {
            "representation_loss": representation_loss,
            "per_example_pred_loss": per_example_pred_loss,
            "per_example_global_agg_loss": per_example_global_agg_loss,
            "per_example_embedding_norm_loss": per_example_embedding_norm_loss,
        }
        return representation_loss, loss_details

    def run_label_propagate(self, verbose=True):
        if verbose:
            print("Propagating labels")
        vote_weights, capital_i = llp_propagate.local_label_propagation_gpu(
            big_m=self.llp_state.big_m_tensor,
            num_labeled=self.llp_params.num_labeled,
            dim_size=self.llp_params.llp_embedding_dim,
            const_k=self.llp_params.llp_const_k,
            const_t=self.llp_params.llp_const_t,
            const_tau=self.llp_params.llp_const_tau,
            chunk_size=self.llp_params.llp_prop_chunk_size,
            device=self.device,
            verbose=verbose,
        )
        true_labels = self.llp_state.all_labels_tensor[:self.llp_params.num_labeled].cpu().numpy()
        pseudolabels, confidence = llp_propagate.compute_pseudolabels(
            true_labels=true_labels,
            vote_weights=vote_weights,
            capital_i=capital_i,
            num_classes=len(self.task.LABELS),
        )
        self.llp_state.all_labels_tensor[self.llp_params.num_labeled:] = \
            torch.from_numpy(pseudolabels).to(self.device)
        self.llp_state.all_label_confidence[self.llp_params.num_labeled:] = \
            torch.from_numpy(confidence).to(self.device)

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
                logits = self.model.forward_batch(batch).logits
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
                logits = self.model.forward_batch(batch).logits
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)

        all_logits = np.concatenate(all_logits, axis=0)
        return all_logits

    def get_train_dataloader(self, train_examples, do_override_labels=True,
                             use_eval_batch_size=False, force_sequential=False,
                             verbose=True):
        dataset_with_metadata = convert_examples_to_dataset(
            examples=train_examples,
            feat_spec=self.rparams.feat_spec,
            tokenizer=self.model_wrapper.tokenizer,
            task=self.task,
            verbose=verbose,
        )

        # Override with pseudolabels
        if do_override_labels:
            override_labels(
                dataset_with_metadata=dataset_with_metadata,
                labels_tensor=self.llp_state.all_labels_tensor.cpu(),
            )

        train_sampler = get_sampler(
            dataset=dataset_with_metadata.dataset,
            local_rank=self.rparams.local_rank,
            force_sequential=force_sequential,
        )
        train_dataloader = DataLoader(
            dataset=dataset_with_metadata.dataset,
            sampler=train_sampler,
            batch_size=self.train_schedule.train_batch_size
            if not use_eval_batch_size else self.rparams.eval_batch_size,
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


def override_labels(dataset_with_metadata, labels_tensor):
    label_column = dataset_with_metadata.get_descriptor_dict()["label_ids"].pos
    tensors = list(dataset_with_metadata.dataset.tensors)
    tensors[label_column] = labels_tensor.cpu()
    dataset_with_metadata.dataset.tensors = tensors
