import numpy as np

from dataclasses import dataclass


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
    steps_per_epoch = int(np.round(num_train_examples / train_batch_size))

    if max_steps > 0:
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
