import torch
import torch.nn as nn

from pytorch_transformers import AdamW, WarmupLinearSchedule


class OptimizerScheduler:
    def __init__(self, optimizer, scheduler):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict, strict=True):
        self.optimizer.load_state_dict(state_dict["optimizer"], strict=strict)
        self.scheduler.load_state_dict(state_dict["scheduler"], strict=strict)


class ModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def create_optimizer(model, learning_rate, t_total, warmup_steps, adam_epsilon=1e-8):
    # Prepare optimizer
    optimized_params = list(model.named_parameters())
    no_decay = [
        'bias', 'LayerNorm.bias', 'LayerNorm.weight',
        'adapter.down_project.weight', 'adapter.up_project.weight',
    ]
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in optimized_params if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
        },
        {
            'params': [p for n, p in optimized_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    optimizer_scheduler = OptimizerScheduler(
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return optimizer_scheduler


def fp16ize(model_wrapper, optimizer_scheduler, fp16_opt_level):
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(
        model_wrapper.model,
        optimizer_scheduler.optimizer,
        opt_level=fp16_opt_level
    )
    model_wrapper.model = model
    optimizer_scheduler.optimizer = optimizer


def parallelize_gpu(model_wrapper):
    model_wrapper.model = torch.nn.DataParallel(model_wrapper.model)


def parallelize_dist(model_wrapper, local_rank):
    model_wrapper.model = torch.nn.parallel.DistributedDataParallel(
        model_wrapper.model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )


def special_model_setup(model_wrapper, optimizer_scheduler,
                        fp16, fp16_opt_level,
                        n_gpu, local_rank):
    if fp16:
        fp16ize(
            model_wrapper=model_wrapper,
            optimizer_scheduler=optimizer_scheduler,
            fp16_opt_level=fp16_opt_level
        )
    if n_gpu > 1:
        parallelize_gpu(model_wrapper=model_wrapper)
    if local_rank != -1:
        parallelize_dist(model_wrapper=model_wrapper, local_rank=local_rank)
