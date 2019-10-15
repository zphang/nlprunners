from dataclasses import dataclass

from pyutils.display import maybe_trange
from pyutils.functional import always_false
from zproto.zlogv1 import BaseZLogger, PRINT_LOGGER

from nlpr.shared.runner import (
    BaseRunner,
    TrainGlobalState,
    save_model_with_metadata,
    compare_steps_max_steps,
)
from nlpr.shared.pycore import ExtendedDataClassMixin
from nlpr.shared.torch_utils import copy_state_dict, CPU_DEVICE


@dataclass
class ValState(ExtendedDataClassMixin):
    score: float
    train_global_state: TrainGlobalState

    def new(self):
        return self.__class__(
            score=self.score,
            train_global_state=self.train_global_state.new(),
        )

    def asdict(self):
        return {
            "score": float(self.score),
            "train_global_state": self.train_global_state.asdict(),
        }


def get_should_save_func(save_every_steps: int):
    if save_every_steps == 0:
        return always_false
    else:
        return lambda tgs: (tgs.global_step + 1) % save_every_steps == 0


def get_should_eval_func(eval_every_steps: int):
    if eval_every_steps == 0:
        return always_false
    else:
        return lambda tgs: (tgs.global_step + 1) % eval_every_steps == 0


class MetaRunner:

    def __init__(self,
                 runner: BaseRunner,
                 train_examples: list, val_examples: list,
                 should_save_func,
                 should_eval_func,
                 output_dir,
                 verbose: bool = True,
                 save_best_model: bool = True,
                 load_best_model: bool = True,
                 log_writer: BaseZLogger = PRINT_LOGGER
                 ):
        self.runner = runner
        self.train_examples = train_examples
        self.val_examples = val_examples
        self.should_save_func = should_save_func
        self.should_eval_func = should_eval_func
        self.output_dir = output_dir
        self.verbose = verbose
        self.save_best_model = save_best_model
        self.load_best_model = load_best_model
        self.log_writer = log_writer

        self.best_val_state = None
        self.best_state_dict = None
        self.val_state_history = []
        self.train_global_state = TrainGlobalState()
        self.full_break = False
        self.single_use_check = False

        # Dependencies
        self.train_schedule = self.runner.train_schedule
        self.model = self.runner.model
        self.device = self.runner.device

    def reset(self):
        self.best_val_state = None
        self.best_state_dict = None
        self.val_state_history = []
        self.train_global_state = TrainGlobalState()
        self.full_break = False
        self.single_use_check = False

    def train_val_save_every(self):
        assert not self.single_use_check
        self.single_use_check = True

        for _ in maybe_trange(
                int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=self.verbose):
            train_dataloader = self.runner.get_train_dataloader(self.train_examples)
            for _ in self.runner.run_train_epoch_context(
                    train_dataloader=train_dataloader,
                    train_global_state=self.train_global_state,
                    verbose=self.verbose):
                self.inject_at_step()

                if self.should_save_func(self.train_global_state):
                    save_model_with_metadata(
                        model=self.model,
                        metadata={},
                        output_dir=self.output_dir,
                        file_name=f"model__{self.train_global_state.global_step}.p",
                    )
                if self.should_eval_func(self.train_global_state):
                    self.eval_save()

                if self.train_schedule.max_steps != -1 and \
                        self.train_global_state.global_step >= self.train_schedule.max_steps:
                    full_break = True

                if compare_steps_max_steps(
                        step=self.train_global_state.global_step,
                        max_steps=self.train_schedule.max_steps):
                    full_break = True

                if self.full_break:
                    break

            if self.full_break:
                break

            self.inject_at_epoch()

        # End of training eval
        self.eval_save()

        if self.load_best_model and self.best_state_dict is not None:
            if self.verbose:
                print("Loading Best")
            self.model.load_state_dict(copy_state_dict(
                state_dict=self.best_state_dict,
                target_device=self.device,
            ))

        return {
            "best_val_state": self.best_val_state,
            "val_state_history": self.val_state_history,
        }

    def inject_at_step(self):
        pass

    def inject_at_epoch(self):
        pass

    def eval_save(self):
        val_result = self.runner.run_val(self.val_examples)
        val_state = ValState(
            score=val_result["metrics"].major,
            train_global_state=self.train_global_state.new(),
        )
        self.log_writer.write_entry("train_val", val_state.asdict())
        self.log_writer.flush()
        if self.best_val_state is None or val_state.score > self.best_val_state.score:
            self.best_val_state = val_state.new()
            self.log_writer.write_entry("train_val_best", self.best_val_state.asdict())
            self.log_writer.flush()
            if self.save_best_model:
                save_model_with_metadata(
                    model=self.model,
                    metadata={
                        "val_state": self.best_val_state.asdict(),
                    },
                    output_dir=self.output_dir,
                    file_name="best_model",
                )
            self.best_state_dict = copy_state_dict(
                state_dict=self.model.state_dict(),
                target_device=CPU_DEVICE,
            )
        self.val_state_history.append(val_state)
