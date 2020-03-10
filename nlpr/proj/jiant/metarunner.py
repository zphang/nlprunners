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
from nlpr.shared.caching import ChunkedFilesDataCache
from nlpr.shared.metarunner_v2 import AbstractMetarunner


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


class JiantMetarunner(AbstractMetarunner):
    def __init__(self,
                 runner: BaseRunner,
                 train_cache: ChunkedFilesDataCache,
                 val_cache: ChunkedFilesDataCache,
                 val_labels_cache: ChunkedFilesDataCache,
                 partial_eval_number: int,
                 should_save_func,
                 should_eval_func,
                 output_dir,
                 verbose: bool = True,
                 save_best_model: bool = True,
                 load_best_model: bool = True,
                 log_writer: BaseZLogger = PRINT_LOGGER
                 ):
        self.runner = runner
        self.train_cache = train_cache
        self.val_cache = val_cache
        self.val_labels_cache = val_labels_cache
        self.partial_eval_number = partial_eval_number
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

    def begin_training(self):
        assert not self.single_use_check
        self.single_use_check = True

    def yield_train_step(self):
        for _ in maybe_trange(
                int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=self.verbose):
            train_dataloader = self.runner.get_train_dataloader(self.train_cache)
            for _ in self.runner.run_train_epoch_context(
                    train_dataloader=train_dataloader,
                    train_global_state=self.train_global_state,
                    verbose=self.verbose):
                self.inject_at_step()
                yield
            self.inject_at_epoch()

    def should_save_model(self) -> bool:
        return self.should_save_func(self.train_global_state)

    def save_model(self):
        save_model_with_metadata(
            model=self.model,
            metadata={},
            output_dir=self.output_dir,
            file_name=f"model__{self.train_global_state.global_step:09d}",
        )

    def should_save_checkpoint(self) -> bool:
        return False

    def save_checkpoint(self):
        raise NotImplementedError()

    def should_eval_model(self) -> bool:
        return self.should_eval_func(self.train_global_state)

    def eval_model(self):
        self.eval_save()

    def should_break_training(self) -> bool:
        if self.train_schedule.max_steps is not None and \
                self.train_schedule.max_steps != -1 and \
                self.train_global_state.global_step >= self.train_schedule.max_steps:
            return True
        elif compare_steps_max_steps(
                step=self.train_global_state.global_step,
                max_steps=self.train_schedule.max_steps):
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

    def inject_at_epoch(self):
        pass

    def eval_save(self):
        val_result = self.runner.run_val(
            val_cache=self.val_cache,
            val_labels_cache=self.val_labels_cache,
            subset_num=self.partial_eval_number,
        )
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
