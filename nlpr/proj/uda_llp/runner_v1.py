from torch.utils.data import DataLoader, SequentialSampler

from nlpr.shared.train_setup import TrainSchedule
from nlpr.shared.runner import (
    convert_examples_to_dataset,
    HybridLoader,
    get_sampler,
)
import nlpr.proj.simple.runner as simple_runner
import nlpr.proj.llp.runner as llp_runner
import nlpr.proj.uda.runner as uda_runner

from pyutils.display import maybe_trange, maybe_tqdm


class UDALLPRunner:
    def __init__(self, task, model_wrapper, optimizer_scheduler, loss_criterion,
                 device,
                 rparams: simple_runner.RunnerParameters,
                 uda_params: uda_runner.UDAParameters,
                 llp_params: llp_runner.LlpParameters,
                 train_schedule: TrainSchedule):
        self.task = task
        self.model_wrapper = model_wrapper
        self.optimizer_scheduler = optimizer_scheduler
        self.loss_criterion = loss_criterion
        self.device = device
        self.rparams = rparams
        self.uda_params = uda_params
        self.llp_params = llp_params
        self.train_schedule = train_schedule

        self.llp_state = None

        # Convenience
        self.model = self.model_wrapper.model

    # LLP
    init_llp_state = llp_runner.LLPRunner.init_llp_state
    create_empty_llp_state = llp_runner.LLPRunner.create_empty_llp_state
    populate_llp_state = llp_runner.LLPRunner.populate_llp_state
    compute_representation_loss = llp_runner.LLPRunner.compute_representation_loss
    run_label_propagate = llp_runner.LLPRunner.run_label_propagate

    # UDA
    get_unsup_dataloaders = uda_runner.UDARunner.get_unsup_dataloaders
    form_dataloader_triplet = uda_runner.UDARunner.form_dataloader_triplet

    # Eval
    run_val = simple_runner.SimpleTaskRunner.run_val
    run_test = simple_runner.SimpleTaskRunner.run_test
    get_eval_dataloader = simple_runner.SimpleTaskRunner.get_eval_dataloader
    complex_propagate = simple_runner.SimpleTaskRunner.complex_backpropagate

    def run_train(self, task_data, verbose=True):
        sup_dataloader = self.get_sup_dataloader(
            train_examples=task_data["sup"]["train"],
            verbose=verbose,
        )

        for _ in maybe_trange(int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=verbose):
            unsup_dataloaders = self.get_unsup_dataloaders(
                sup_dataloader=sup_dataloader,
                task_data=task_data,
            )
            dataloader_triplet = self.form_dataloader_triplet(
                sup_dataloader=sup_dataloader,
                unsup_orig_loader=unsup_dataloaders.unsup_orig,
                unsup_aug_loader=unsup_dataloaders.unsup_aug,
            )
            self.run_train_epoch(dataloader_triplet, verbose=verbose)

    def get_sup_dataloader(self, train_examples, do_override_labels=True,
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
            llp_runner.override_labels(
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
