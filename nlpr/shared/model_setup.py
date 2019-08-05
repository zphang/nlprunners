import torch
import torch.nn as nn

import pytorch_transformers


def simple_model_setup(model_type, model_class_spec, config_path, tokenizer_path, task):
    config = model_class_spec.config_class.from_json_file(config_path)
    config.num_labels = len(task.LABELS)
    model = model_class_spec.model_class(config)
    if "-cased" in model_type:
        do_lower_case = False
    elif "-uncased" in model_type:
        do_lower_case = True
    else:
        raise RuntimeError(model_type)
    tokenizer = model_class_spec.tokenizer_class.from_pretrained(
        tokenizer_path, do_lower_case=do_lower_case,
    )
    return ModelWrapper(
        model=model,
        tokenizer=tokenizer
    )


def load_model(model, state_dict, max_miss_fraction=0.9, verbose=True):
    missed, unused = model.load_state_dict(state_dict, strict=False)
    total_mismatched = len(missed) + len(unused)
    total_params = len(model.state_dict())
    if verbose:
        print(f"Missed {len(missed)}:")
        for pname in missed:
            print(f"  {pname}")
        print(f"Unused {len(unused)}:")
        for pname in unused:
            print(f"  {pname}")
    if total_mismatched / total_params > 1 - max_miss_fraction:
        raise RuntimeError(f"Mismatched {total_mismatched} out of {total_params} parameters")


class OptimizerScheduler:
    def __init__(self, optimizer, scheduler):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler

    def step(self):
        # Scheduler updates first
        self.scheduler.step()
        self.optimizer.step()

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
    optimizer = pytorch_transformers.AdamW(
        optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon
    )
    scheduler = pytorch_transformers.WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=t_total
    )
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
