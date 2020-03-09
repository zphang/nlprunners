import abc
import numpy as np

from typing import Union, Optional


class BaseMultiTaskScheduler(metaclass=abc.ABCMeta):
    def __init__(self,
                 task_dict: dict, rng: Union[int, np.random.RandomState, None]
                 ):
        self.task_dict = task_dict
        if isinstance(rng, int) or rng is None:
            rng = np.random.RandomState(rng)
        self.rng = rng

    def pop(self):
        raise NotImplementedError()

    def iter(self):
        yield self.pop()


class UniformMultiTaskScheduler(BaseMultiTaskScheduler):

    def pop(self):
        task_name = self.rng.choice(self.task_dict)
        return task_name, self.task_dict[task_name]


class ProportionalMultiTaskScheduler(BaseMultiTaskScheduler):

    def __init__(self,
                 task_dict: dict, rng: Union[int, np.random.RandomState],
                 task_to_examples_dict: dict,
                 ):
        super().__init__(task_dict=task_dict, rng=rng)
        self.task_to_examples_dict = task_to_examples_dict
        self.task_names = list(task_to_examples_dict.keys())
        self.task_num_examples = np.array([task_to_examples_dict[k] for k in self.task_names])
        self.task_p = self.task_num_examples / self.task_num_examples.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]


class TemperatureMultiTaskScheduler(BaseMultiTaskScheduler):

    def __init__(self,
                 task_dict: dict, rng: Union[int, np.random.RandomState],
                 task_to_examples_dict: dict,
                 temperature: float,
                 examples_cap: Optional[int],
                 ):
        super().__init__(task_dict=task_dict, rng=rng)
        self.task_to_examples_dict = task_to_examples_dict
        self.temperature = temperature
        self.examples_cap = examples_cap
        self.task_names = list(task_to_examples_dict.keys())
        self.task_num_examples = np.array([task_to_examples_dict[k] for k in self.task_names])
        raw_n = self.task_num_examples.clip(max=examples_cap) ** (1/self.temperature)
        self.task_p = raw_n / raw_n.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]


def create_scheduler(init_dict: dict,
                     task_dict: dict,
                     task_to_examples_dict: dict,
                     rng=None) -> BaseMultiTaskScheduler:
    scheduler_type = init_dict["scheduler_type"]
    if scheduler_type == "UniformMultiTaskScheduler":
        assert len(init_dict) == 1
        return UniformMultiTaskScheduler(task_dict=task_dict, rng=rng)
    elif scheduler_type == "ProportionalMultiTaskScheduler":
        assert len(init_dict) == 1
        return ProportionalMultiTaskScheduler(
            task_dict=task_dict,
            rng=rng,
            task_to_examples_dict=task_to_examples_dict,
        )
    elif scheduler_type == "TemperatureMultiTaskScheduler":
        assert len(init_dict) == 3
        return TemperatureMultiTaskScheduler(
            task_dict=task_dict,
            rng=rng,
            task_to_examples_dict=task_to_examples_dict,
            temperature=init_dict["temperature"],
            examples_cap=init_dict["examples_cap"],
        )
    else:
        raise KeyError(scheduler_type)
