import os
from dataclasses import dataclass

from pyutils.functional import always_false
from zproto.zlogv1 import BaseZLogger, PRINT_LOGGER

from nlpr.shared.runner import (
    save_model_with_metadata,
    compare_steps_max_steps,
)
from nlpr.shared.pycore import ExtendedDataClassMixin
from nlpr.shared.torch_utils import copy_state_dict, CPU_DEVICE
from nlpr.shared.metarunner_v2 import AbstractMetarunner
import nlpr.proj.jiant.runner as jiant_runner
import nlpr.proj.jiant.components.task_sampler as jiant_task_sampler


@dataclass
class ValState(ExtendedDataClassMixin):
    score: float
    train_state: jiant_runner.TrainState

    def new(self):
        return self.__class__(
            score=self.score,
            train_state=self.train_state.new(),
        )

    def to_dict(self):
        return {
            "score": float(self.score),
            "train_state": self.train_state.to_dict(),
        }


def get_should_save_func(save_every_steps: int):
    if save_every_steps == 0:
        return always_false
    else:
        return lambda tgs: (tgs.global_steps + 1) % save_every_steps == 0


def get_should_eval_func(eval_every_steps: int):
    if eval_every_steps == 0:
        return always_false
    else:
        return lambda tgs: (tgs.global_steps + 1) % eval_every_steps == 0


def get_should_save_checkpoint_func(save_checkpoint_every_steps: int):
    if save_checkpoint_every_steps == 0:
        return always_false
    else:
        return lambda tgs: (tgs.global_steps + 1) % save_checkpoint_every_steps == 0


class JiantMetarunner(AbstractMetarunner):
    def __init__(self,
                 runner: jiant_runner.JiantRunner,
                 should_save_func,
                 should_eval_func,
                 should_save_checkpoint_func,
                 checkpoint_saver,
                 output_dir,
                 verbose: bool = True,
                 save_best_model: bool = True,
                 load_best_model: bool = True,
                 log_writer: BaseZLogger = PRINT_LOGGER
                 ):
        self.runner = runner
        self.should_save_func = should_save_func
        self.should_eval_func = should_eval_func
        self.should_save_checkpoint_func = should_save_checkpoint_func
        self.checkpoint_saver = checkpoint_saver
        self.output_dir = output_dir
        self.verbose = verbose
        self.save_best_model = save_best_model
        self.load_best_model = load_best_model
        self.log_writer = log_writer

        self.best_val_state = None
        self.best_state_dict = None
        self.val_state_history = []
        self.train_state = None
        self.full_break = False
        self.single_use_check = False

        self.model = self.runner.model
        self.device = self.runner.device
        self.global_train_config = self.runner.jiant_task_container.global_train_config

    def begin_training(self):
        assert not self.single_use_check
        self.single_use_check = True

    def yield_train_step(self):
        if self.train_state is None:
            # Fresh run
            train_iterator = self.runner.run_train_context(verbose=self.verbose)
        else:
            train_iterator = self.runner.resume_train_context(
                train_state=self.train_state,
                verbose=self.verbose,
            )
        for train_state in train_iterator:
            self.train_state = train_state
            self.inject_at_step()
            yield

    def should_save_model(self) -> bool:
        return self.should_save_func(self.train_state)

    def save_model(self):
        save_model_with_metadata(
            model=self.model,
            metadata={},
            output_dir=self.output_dir,
            file_name=f"model__{self.train_state.global_steps:09d}",
        )

    def should_save_checkpoint(self) -> bool:
        return self.should_save_checkpoint_func(self.train_state)

    def save_checkpoint(self):
        runner_state = self.runner.get_runner_state()
        metarunner_state = self.get_state()
        print("Saving State")
        self.checkpoint_saver.save(runner_state=runner_state, metarunner_state=metarunner_state)

    def get_state(self):
        return {
            "best_val_state": self.best_val_state,
            "best_state_dict": self.best_state_dict,
            "train_state": self.train_state,
        }

    def load_state(self, metarunner_state):
        self.best_val_state = metarunner_state["best_val_state"]
        self.best_state_dict = metarunner_state["best_state_dict"]
        self.train_state = metarunner_state["train_state"]

    def should_eval_model(self) -> bool:
        return self.should_eval_func(self.train_state)

    def eval_model(self):
        self.eval_save()

    def should_break_training(self) -> bool:
        if self.global_train_config.max_steps is not None and \
                self.global_train_config.max_steps != -1 and \
                self.train_state.global_steps >= self.global_train_config.max_steps:
            return True
        elif compare_steps_max_steps(
                step=self.train_state.global_steps,
                max_steps=self.global_train_config.max_steps):
            return True
        else:
            return False

    def done_training(self):
        self.eval_save()
        if self.load_best_model and self.best_state_dict is not None:
            if self.verbose:
                print("Loading Best")
            self.model.load_state_dict(copy_state_dict(
                state_dict=self.best_state_dict,
                target_device=self.device,
            ))

    def returned_result(self):
        return {
            "best_val_state": self.best_val_state,
            "val_state_history": self.val_state_history,
        }

    # ======================== #

    def inject_at_step(self):
        pass

    def eval_save(self):
        val_results_dict = self.runner.run_val(use_subset=True)
        aggregated_major = jiant_task_sampler.compute_aggregate_major_metrics_from_results_dict(
            metrics_aggregator=self.runner.jiant_task_container.metrics_aggregator,
            results_dict=val_results_dict,
        )
        val_state = ValState(
            score=float(aggregated_major),
            train_state=self.train_state.new(),
        )
        self.log_writer.write_entry("train_val", val_state.to_dict())
        self.log_writer.flush()
        if self.best_val_state is None or val_state.score > self.best_val_state.score:
            self.best_val_state = val_state.new()
            self.log_writer.write_entry("train_val_best", self.best_val_state.to_dict())
            self.log_writer.flush()
            if self.save_best_model:
                save_model_with_metadata(
                    model=self.model,
                    metadata={
                        "val_state": self.best_val_state.to_dict(),
                    },
                    output_dir=self.output_dir,
                    file_name="best_model",
                )
            self.best_state_dict = copy_state_dict(
                state_dict=self.model.state_dict(),
                target_device=CPU_DEVICE,
            )
        self.val_state_history.append(val_state)
