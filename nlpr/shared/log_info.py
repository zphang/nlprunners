import os
import time


def print_args(args):
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))


def print_trainable_params(model):
    print("TRAINABLE PARAMS:")
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            print("    {}  {}".format(param_name, tuple(param.shape)))


class FakeTensorboardWriter:
    @classmethod
    def add_scalar(cls, tag, scalar_value, global_step=None):
        g_step_format = f"({global_step})" if global_step else ""
        print(f"TB: [{tag}]: {scalar_value} {g_step_format}")

    @classmethod
    def add_scalars(cls, main_tag, tag_scalar_dict, global_step=None):
        g_step_format = f"({global_step})" if global_step else ""
        print(f"TB: [{main_tag}] {g_step_format}")
        for k, v in tag_scalar_dict:
            print(f"    [{k}] {v}")

    @classmethod
    def flush(cls):
        pass


class SilentTensorboardWriter:
    @classmethod
    def add_scalar(cls, tag, scalar_value, global_step=None):
        pass

    @classmethod
    def add_scalars(cls, main_tag, tag_scalar_dict, global_step=None):
        pass

    @classmethod
    def flush(cls):
        pass


"""
# from torch.utils.tensorboard import SummaryWriter
# Todo: optionally use logger

def get_tb_writer_with_unix_time(log_dir):
    return SummaryWriter(os.path.join(log_dir, f"{int(time.time())}"))


def simple_setup_tensorboard(use_tensorboard, output_dir):
    if use_tensorboard:
        return get_tb_writer_with_unix_time(output_dir)
    else:
        return SilentTensorboardWriter
"""
