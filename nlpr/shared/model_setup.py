import torch
import torch.nn as nn

import pytorch_transformers

from nlpr.shared.model_resolution import ModelArchitectures
from nlpr.tasks.lib.shared import TaskTypes


class ModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def simple_model_setup(model_type, model_class_spec, config_path, tokenizer_path, task):
    config = model_class_spec.config_class.from_json_file(config_path)
    if task.TASK_TYPE == TaskTypes.CLASSIFICATION:
        config.num_labels = len(task.LABELS)
    elif task.TASK_TYPE == TaskTypes.REGRESSION:
        config.num_labels = 1
    elif task.TASK_TYPE == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        config.num_labels = len(task.LABELS)
        config.num_spans = 2  # todo: this is hardcoded
    else:
        raise KeyError(task.TASK_TYPE)

    model = model_class_spec.model_class(config)
    model_arch = ModelArchitectures.from_model_type(model_type)
    if model_arch in [ModelArchitectures.BERT]:
        if "-cased" in model_type:
            do_lower_case = False
        elif "-uncased" in model_type:
            do_lower_case = True
        else:
            raise RuntimeError(model_type)
    elif model_arch in [
            ModelArchitectures.XLNET, ModelArchitectures.XLM, ModelArchitectures.ROBERTA]:
        do_lower_case = False
    else:
        raise RuntimeError(model_type)
    print(do_lower_case)
    tokenizer = model_class_spec.tokenizer_class.from_pretrained(
        tokenizer_path, do_lower_case=do_lower_case,
    )
    return ModelWrapper(
        model=model,
        tokenizer=tokenizer
    )


def safe_load_model(model, state_dict, max_miss_fraction=0.9, verbose=True):
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


def simple_load_model(model, state_dict, model_load_mode, verbose=True):
    if model_load_mode == "strict":
        model.load_state_dict(state_dict)
    elif model_load_mode == "safe":
        safe_load_model(
            model=model,
            state_dict=state_dict,
            verbose=verbose,
        )
    elif model_load_mode == "base_weights":
        model.load_state_dict(load_model_base_weights(
            model=model,
            state_dict=state_dict,
        ))
    else:
        raise KeyError(model_load_mode)


def load_model_base_weights(model, state_dict):
    arch = ModelArchitectures.from_ptt_model(model)
    if arch == ModelArchitectures.BERT:
        new_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("classifier.")
        }
    else:
        raise NotImplementedError()
    return new_state_dict


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


def create_optimizer(model, learning_rate, t_total, warmup_steps, warmup_proportion,
                     adam_epsilon=1e-8, verbose=False):
    create_optimizer_from_params(
        named_parameters=list(model.named_parameters()),
        learning_rate=learning_rate,
        t_total=t_total,
        warmup_steps=warmup_steps,
        warmup_proportion=warmup_proportion,
        adam_epsilon=adam_epsilon,
        verbose=False,
    )


def create_optimizer_from_params(named_parameters, learning_rate, t_total, warmup_steps, warmup_proportion,
                                 adam_epsilon=1e-8, verbose=False):
    # Prepare optimizer
    no_decay = [
        'bias', 'LayerNorm.bias', 'LayerNorm.weight',
        'adapter.down_project.weight', 'adapter.up_project.weight',
    ]
    if verbose:
        print("No optimizer decay for:")
        for n, p in named_parameters:
            if any(nd in n for nd in no_decay):
                print(f"  {n}")

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
        },
        {
            'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    optimizer = pytorch_transformers.AdamW(
        optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon
    )
    warmup_steps = resolve_warmup_steps(
        t_total=t_total, warmup_steps=warmup_steps,
        warmup_proportion=warmup_proportion,
    )
    scheduler = pytorch_transformers.WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=t_total
    )
    optimizer_scheduler = OptimizerScheduler(
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return optimizer_scheduler


def resolve_warmup_steps(t_total, warmup_steps, warmup_proportion):
    if warmup_steps is None and warmup_proportion is None:
        raise RuntimeError()
    elif warmup_steps is not None and warmup_proportion is not None:
        raise RuntimeError()
    elif warmup_steps is None and warmup_proportion is not None:
        return warmup_proportion * t_total
    elif warmup_steps is not None and warmup_proportion is None:
        return warmup_steps
    else:
        raise RuntimeError()


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
