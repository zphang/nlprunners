import os

from nlpr.tasks.lib.anli import AnliTask
from nlpr.tasks.lib.amazon import AmazonPolarityTask
from nlpr.tasks.lib.commitmentbank import CommitmentBankTask
from nlpr.tasks.lib.copa import CopaTask
from nlpr.tasks.lib.multirc import MultiRCTask
from nlpr.tasks.lib.rte import RteTask
from nlpr.tasks.lib.wic import WiCTask
from nlpr.tasks.lib.wsc import WSCTask
from nlpr.tasks.lib.yelp import YelpPolarityTask
from nlpr.tasks.lib.imdb import IMDBTask
from nlpr.tasks.lib.maskedwiki import MaskedWikiTask
from nlpr.tasks.lib.mnli import MnliTask
from nlpr.tasks.lib.mrpc import MrpcTask
from nlpr.tasks.lib.cola import ColaTask
from nlpr.tasks.lib.boolq import BoolQTask
from nlpr.tasks.lib.record import ReCoRDTask
from nlpr.tasks.lib.qqp import QqpTask
from nlpr.tasks.lib.qnli import QnliTask
from nlpr.tasks.lib.snli import SnliTask
from nlpr.tasks.lib.squad import SquadTask
from nlpr.tasks.lib.sst import SstTask
from nlpr.tasks.lib.stsb import StsbTask
from nlpr.tasks.lib.wnli import WnliTask
from nlpr.tasks.lib.templates.shared import Task

from pyutils.io import read_json


TASK_DICT = {
    "anli": AnliTask,
    "cb": CommitmentBankTask,
    "copa": CopaTask,
    "mrc": MultiRCTask,
    "rte": RteTask,
    "wic": WiCTask,
    "wsc": WSCTask,
    "yelp": YelpPolarityTask,
    "amazon": AmazonPolarityTask,
    "mnli": MnliTask,
    "imdb": IMDBTask,
    "mrpc": MrpcTask,
    "cola": ColaTask,
    "boolq": BoolQTask,
    "qqp": QqpTask,
    "qnli": QnliTask,
    "record": ReCoRDTask,
    "snli": SnliTask,
    "squad_v1": SquadTask,
    "squad_v2": SquadTask,
    "sst": SstTask,
    "stsb": StsbTask,
    "wnli": WnliTask,
    "masked_wiki": MaskedWikiTask,
}


def get_task(task_name, data_dir):
    task_name = task_name.lower()
    task_class = TASK_DICT[task_name]
    return task_class(task_name, data_dir)


def get_task_class(task_name):
    task_class = TASK_DICT[task_name]
    assert issubclass(task_class, Task)
    return task_class


def create_task_from_config(config: dict, base_path=None, verbose=False):
    task_class = get_task_class(config["task"])
    for k in config["paths"].keys():
        path = config["paths"][k]
        if not os.path.isabs(path):
            assert base_path
            config["paths"][k] = os.path.join(base_path, path)
    task_kwargs = config.get("kwargs", {})
    if verbose:
        print(task_class.__name__)
        for k, v in config["paths"].items():
            print(f"  [{k}]: {v}")
    return task_class(name=config["task"], path_dict=config["paths"], **task_kwargs)


def create_task_from_config_path(config_path: str, verbose=False):
    return create_task_from_config(
        read_json(config_path),
        base_path=os.path.split(config_path)[0],
        verbose=verbose,
    )
