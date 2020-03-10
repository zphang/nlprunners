from dataclasses import dataclass
from typing import Dict


import nlpr.shared.pycore as pycore
import nlpr.tasks as tasks
import nlpr.proj.jiant.components.task_sampler as jiant_task_sampler
import nlpr.shared.train_setup as shared_train_setup
import nlpr.shared.caching as caching
from nlpr.constants import PHASE


@dataclass
class JiantTaskContainer:
    task_dict: Dict[str, tasks.Task]
    task_sampler: jiant_task_sampler.BaseMultiTaskSampler
    train_schedule: shared_train_setup.TrainSchedule
    task_cache_dict: Dict


@dataclass
class RunnerParameters:
    local_rank: int
    n_gpu: int
    fp16: bool
    learning_rate: float
    max_grad_norm: float


@dataclass
class GlobalTrainConfig(pycore.ExtendedDataClassMixin):
    max_steps: int
    warmup_steps: int


@dataclass
class TaskSpecificConfig(pycore.ExtendedDataClassMixin):
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int


def create_task_dict(task_config_dict: dict,
                     verbose: bool = True) -> Dict[str, tasks.Task]:
    return {
        task_name: tasks.create_task_from_config_path(
            config_path=task_config_path,
            verbose=verbose,
        )
        for task_name, task_config_path in task_config_dict.items()
    }


def create_task_cache_dict(task_cache_config_dict: Dict) -> Dict:
    task_cache_dict = {}
    for task_name, task_cache_config in task_cache_config_dict.items():
        single_task_cache_dict = {}
        for phase in [PHASE.TRAIN, PHASE.VAL, PHASE.TEST]:
            if phase in task_cache_config_dict:
                single_task_cache_dict[phase] = caching.ChunkedFilesDataCache(
                    task_cache_config_dict[phase],
                )
        task_cache_dict[task_name] = single_task_cache_dict
    return task_cache_dict


def get_num_train_examples(task_cache_dict: Dict) -> Dict[int]:
    return {
        task_name: len(single_task_cache_dict[PHASE.TRAIN])
        for task_name, single_task_cache_dict in task_cache_dict.items()
    }


def create_jiant_task_container(task_config_dict: Dict,
                                sampler_config: Dict,
                                task_cache_config_dict: Dict):
    task_dict = create_task_dict(task_config_dict=task_config_dict)
    task_cache_dict = create_task_cache_dict(task_cache_config_dict=task_cache_config_dict)
    num_train_examples_dict = get_num_train_examples(
        task_cache_dict=task_cache_dict,
    )
    task_sampler = jiant_task_sampler.create_task_sampler(
        sampler_config=sampler_config,
        task_dict=task_dict,
        task_to_examples_dict=num_train_examples_dict,
    )
    train_schedule = shared_train_setup.TrainSchedule(
        train_batch_size
    )
