import os

import torch

from dataclasses import dataclass

from pyutils.display import maybe_tqdm
from pyutils.functional import always_false
from zproto.zlogv1 import BaseZLogger, PRINT_LOGGER

from nlpr.shared.runner import (
    BaseRunner,
    TrainGlobalState,
    save_model_with_metadata,
)
from nlpr.shared.pycore import ExtendedDataClassMixin


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


def train_val_save_every(runner: BaseRunner,
                         train_examples: list, val_examples: list,
                         should_save_func,
                         should_eval_func,
                         output_dir,
                         verbose: bool = True,
                         save_best_model: bool = True,
                         load_best_model: bool = True,
                         log_writer: BaseZLogger = PRINT_LOGGER):
    if load_best_model:
        assert save_best_model

    train_global_state = TrainGlobalState()
    best_val_state = None
    val_state_history = []
    for _ in maybe_tqdm(
            int(runner.train_schedule.num_train_epochs), desc="Epoch", verbose=verbose):
        train_dataloader = runner.get_train_dataloader(train_examples)
        for _ in runner.run_train_epoch_context(
                train_dataloader=train_dataloader,
                train_global_state=train_global_state,
                verbose=verbose):
            if should_save_func(train_global_state):
                save_model_with_metadata(
                    model=runner.model,
                    metadata={},
                    output_dir=output_dir,
                    file_name=f"model__{train_global_state.global_step}.p",
                )
            if should_eval_func(train_global_state):
                val_result = runner.run_val(val_examples)
                val_state = ValState(
                    score=val_result["metrics"]["major"],
                    train_global_state=train_global_state.new(),
                )
                log_writer.write_entry("train_val", val_state.asdict()  )
                log_writer.flush()
                if best_val_state is None or val_state.score > best_val_state.score:
                    best_val_state = val_state.new()
                    log_writer.write_entry("train_val_best", best_val_state.asdict())
                    log_writer.flush()
                    save_model_with_metadata(
                        model=runner.model,
                        metadata={
                            "val_state": best_val_state.as_dict(),
                        },
                        output_dir=output_dir,
                        file_name="best_model.p",
                    )
                val_state_history.append(val_state)

    if load_best_model:
        runner.model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.p")))

    return {
        "best_val_state": best_val_state,
        "val_state_history": val_state_history,
    }
