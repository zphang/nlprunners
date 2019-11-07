import numpy as np

import torch.nn as nn

from dataclasses import dataclass
from nlpr.tasks.lib.shared import TaskTypes


@dataclass
class TrainSchedule:
    train_batch_size: int
    max_steps: int
    num_train_epochs: float
    t_total: int
    gradient_accumulation_steps: int


def get_train_schedule(num_train_examples,
                       max_steps, num_train_epochs,
                       gradient_accumulation_steps, per_gpu_train_batch_size, n_gpu):
    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    steps_per_epoch = int(np.ceil(num_train_examples / train_batch_size))

    if max_steps is not None and max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps // (steps_per_epoch // gradient_accumulation_steps) + 1
    else:
        t_total = steps_per_epoch // gradient_accumulation_steps * num_train_epochs
        num_train_epochs = num_train_epochs

    return TrainSchedule(
        train_batch_size=train_batch_size,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        t_total=t_total,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


def resolve_loss_function(task_type: TaskTypes):
    # maybe move this function
    if task_type == TaskTypes.CLASSIFICATION:
        return nn.CrossEntropyLoss()
    elif task_type == TaskTypes.REGRESSION:
        return nn.MSELoss()
    elif task_type == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        return nn.CrossEntropyLoss()
    if task_type == TaskTypes.MULTIPLE_CHOICE:
        return nn.CrossEntropyLoss()
    else:
        raise KeyError(task_type)


def maybe_subsample_train(train_examples, train_examples_number, train_examples_fraction):
    if train_examples_fraction == 1.0:
        train_examples_fraction = None
    if train_examples_number is None and train_examples_fraction is None:
        return train_examples, None
    elif train_examples_number is None and train_examples_fraction is not None:
        return random_sample_fraction(train_examples, train_examples_fraction, replace=False)
    elif train_examples_number is not None and train_examples_fraction is None:
        # Cap at train dataset size
        train_examples_number = min(len(train_examples), train_examples_number)
        return random_sample(train_examples, train_examples_number, replace=False)
    else:
        raise RuntimeError


def random_sample_fraction(ls, fraction, replace=True):
    return random_sample(
        ls=ls,
        size=int(np.floor(len(ls) * fraction)),
        replace=replace,
    )


def random_sample(ls, size, replace=True):
    indices = [
        int(i)
        for i in np.random.choice(range(len(ls)), size=size, replace=replace)
    ]
    return [ls[i] for i in indices], indices
