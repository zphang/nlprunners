import os

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
}

DEFAULT_FOLDER_NAMES = {
    "cb": "CB",
    "copa": "COPA",
    "mrc": "MultiRC",
    "rte": "RTE",
    "wic": "WiC",
    "wsc": "WSC",
    "yelp_polarity": "YelpPolarity",
    "amzn_polarity": "AmazonPolarity",
    "mrpc": "MRPC",
}


def get_task(task_name, data_dir):
    task_name = task_name.lower()
    task_class = TASK_DICT[task_name]
    if data_dir is None:
        data_dir = os.path.join(os.environ["TASK_DIR"], DEFAULT_FOLDER_NAMES[task_name])
    return task_class(task_name, data_dir)
