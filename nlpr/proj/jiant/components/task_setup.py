from dataclasses import dataclass
from typing import Dict

import pyutils.io as io

import nlpr.shared.pycore as pycore
import nlpr.tasks as tasks
import nlpr.proj.jiant.components.task_sampler as jiant_task_sampler
import nlpr.shared.caching as caching
from nlpr.constants import PHASE


@dataclass
class TaskSpecificConfig(pycore.ExtendedDataClassMixin):
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    eval_subset_num: int


@dataclass
class GlobalTrainConfig(pycore.ExtendedDataClassMixin):
    max_steps: int
    warmup_steps: int


@dataclass
class JiantTaskContainer:
    task_dict: Dict[str, tasks.Task]
    task_sampler: jiant_task_sampler.BaseMultiTaskSampler
    task_cache_dict: Dict
    global_train_config: GlobalTrainConfig
    task_specific_configs: Dict[str, TaskSpecificConfig]
    metrics_aggregator: jiant_task_sampler.BaseMetricAggregator


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
        for phase in [PHASE.TRAIN, PHASE.VAL, "val_labels", PHASE.TEST]:
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


def create_task_specific_configs(task_specific_configs_dict) -> Dict[str, TaskSpecificConfig]:
    task_specific_configs = {}
    for k, v in task_specific_configs_dict.items():
        if isinstance(v, dict):
            v = TaskSpecificConfig.from_dict(v)
        elif isinstance(v, TaskSpecificConfig):
            pass
        else:
            raise TypeError(type(v))
        task_specific_configs[k] = v
    return task_specific_configs


def create_jiant_task_container(task_config_dict: Dict,
                                task_cache_config_dict: Dict,
                                sampler_config: Dict,
                                global_train_config_dict: Dict,
                                task_specific_configs_dict: Dict,
                                metric_aggregator_config: Dict) \
        -> JiantTaskContainer:
    task_dict = create_task_dict(
        task_config_dict=task_config_dict,
    )
    task_cache_dict = create_task_cache_dict(
        task_cache_config_dict=task_cache_config_dict,
    )
    num_train_examples_dict = get_num_train_examples(
        task_cache_dict=task_cache_dict,
    )
    task_sampler = jiant_task_sampler.create_task_sampler(
        sampler_config=sampler_config,
        task_dict=task_dict,
        task_to_examples_dict=num_train_examples_dict,
    )
    global_train_config = GlobalTrainConfig.from_dict(global_train_config_dict)
    task_specific_config = create_task_specific_configs(
        task_specific_configs_dict=task_specific_configs_dict,
    )
    metric_aggregator = jiant_task_sampler.create_metric_aggregator(
        metric_aggregator_config=metric_aggregator_config,
    )
    return JiantTaskContainer(
        task_dict=task_dict,
        task_sampler=task_sampler,
        global_train_config=global_train_config,
        task_cache_dict=task_cache_dict,
        task_specific_configs=task_specific_config,
        metrics_aggregator=metric_aggregator,
    )


def create_jiant_task_container_from_paths(task_config_dict_path: Dict,
                                           task_cache_config_dict_path: Dict,
                                           sampler_config_path: Dict,
                                           global_train_config_dict_path: Dict,
                                           task_specific_configs_dict_path: Dict,
                                           metric_aggregator_config_path: Dict) \
        -> JiantTaskContainer:
    return create_jiant_task_container(
        task_config_dict=io.read_json(task_config_dict_path),
        task_cache_config_dict=io.read_json(task_cache_config_dict_path),
        sampler_config=io.read_json(sampler_config_path),
        global_train_config_dict=io.read_json(global_train_config_dict_path),
        task_specific_configs_dict=io.read_json(task_specific_configs_dict_path),
        metric_aggregator_config=io.read_json(metric_aggregator_config_path),
    )
