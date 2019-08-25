import math

from dataclasses import dataclass

from pyutils.display import maybe_tqdm, maybe_trange

from nlpr.proj.simple.runner import SimpleTaskRunner
from nlpr.shared.modeling import forward_batch_basic
from nlpr.shared.runner import TrainEpochState
from nlpr.shared.torch_utils import compute_pred_entropy_clean


@dataclass
class NTrainingTrainGlobalState:
    epoch: int = 0
    global_step_float: float = 0


class NTrainingSingleRunner(SimpleTaskRunner):

    def run_train_epoch(self, train_dataloader, train_global_state, verbose=True):
        assert isinstance(train_global_state, NTrainingTrainGlobalState)
        for _ in self.run_train_epoch_context(
                train_dataloader=train_dataloader,
                train_global_state=train_global_state,
                verbose=verbose):
            pass

    def run_train_epoch_context(self, train_dataloader, train_global_state, verbose=True):
        train_epoch_state = TrainEpochState()

        # We shrink the step size such that each epoch takes the same number of optimizer steps,
        #   regardless of how many examples there actually are
        labeled_steps_per_epoch = self.train_schedule.t_total / self.train_schedule.num_train_epochs
        step_size = labeled_steps_per_epoch / len(train_dataloader)

        for step, (batch, batch_metadata) in enumerate(
                maybe_tqdm(train_dataloader, desc="Training", verbose=verbose)):
            self.run_train_step(
                step=step,
                batch=batch,
                train_epoch_state=train_epoch_state,
                train_global_state=train_global_state,
                step_size=step_size,
            )
            yield step, batch, train_epoch_state

    def run_train_step(self, step, batch, train_epoch_state, train_global_state, step_size=1):
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
            if add_crosses_int(train_global_state.global_step, step_size):
                self.optimizer_scheduler.scheduler.step()

            train_epoch_state.global_step += step_size
            train_global_state.global_step += step_size
            self.optimizer_scheduler.optimizer.step()
            self.model.zero_grad()

        self.log_writer.write_entry("loss_train", {
            "epoch": train_global_state.epoch,
            "epoch_step": train_epoch_state.global_step,
            "global_step": train_global_state.global_step,
            "loss_val": loss_val,
            "pred_entropy": compute_pred_entropy_clean(logits)
        })


class NTrainingOverallRunner:
    def __init__(self, runners_ls, labeled_train_examples, unlabeled_task_data):
        self.runners_ls = runners_ls
        self.labeled_train_examples = labeled_train_examples
        self.unlabeled_task_data = unlabeled_task_data

    def train_phase(self):
        for runner in self.runners_ls:
            runner.run_train_epoch


def add_crosses_int(a, b):
    return math.floor(a + b) > math.floor(a)
