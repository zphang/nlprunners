from nlpr.tasks.lib.commitmentbank import CommitmentBankTask
from nlpr.tasks.lib.copa import CopaTask
from nlpr.tasks.lib.multirc import MultiRCTask
from nlpr.tasks.lib.rte import RteTask
from nlpr.tasks.lib.wic import WiCTask
from nlpr.tasks.lib.wsc import WSCTask
from nlpr.tasks.lib.yelp import YelpPolarityTask
from nlpr.tasks.lib.amazon import AmazonPolarityTask
from nlpr.tasks.lib.mnli import MnliTask
from nlpr.tasks.lib.imdb import IMDBTask
from nlpr.tasks.lib.mrpc import MrpcTask
from nlpr.tasks.lib.cola import ColaTask
from nlpr.tasks.lib.shared import Task

from pyutils.io import read_json


TASK_DICT = {
    "cb": CommitmentBankTask,
    "copa": CopaTask,
    "mrc": MultiRCTask,
    "rte": RteTask,
    "wic": WiCTask,
    "wsc": WSCTask,
    "yelp_polarity": YelpPolarityTask,
    "amzn_polarity": AmazonPolarityTask,
    "mnli": MnliTask,
    "imdb": IMDBTask,
    "mrpc": MrpcTask,
    "cola": ColaTask,
}


def get_task(task_name, data_dir):
    task_name = task_name.lower()
    task_class = TASK_DICT[task_name]
    return task_class(task_name, data_dir)


def get_task_class(task_name):
    task_class = TASK_DICT[task_name]
    assert issubclass(task_class, Task)
    return task_class


def create_task_from_config(config: dict):
    task_class = get_task_class(config["task"])
    return task_class(name=config["task"], path_dict=config["paths"])


def create_task_from_config_path(config_path: str):
    return create_task_from_config(read_json(config_path))
