import os
import numpy as np
import torch
import tqdm

from dataclasses import dataclass
from typing import Sequence

import zconf

import nlpr.shared.initialization as initialization
import nlpr.shared.model_setup as model_setup
import nlpr.shared.model_resolution as model_resolution
import nlpr.shared.modeling.models as shared_models
import nlpr.tasks as tasks
import nlpr.shared.runner as shared_runner
import nlpr.shared.caching as caching
import nlpr.proj.weight_study.cka as cka
from torch.utils.data.dataloader import DataLoader


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_config_path = zconf.attr(type=str, required=True)
    task_cache_path = zconf.attr(type=str, default=None)
    indices_path = zconf.attr(type=str, default=None)
    output_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_tokenizer_path = zconf.attr(default=None, type=str)

    model_a_path = zconf.attr(type=str, required=True)
    model_b_path = zconf.attr(type=str, required=True)

    # === Running Setup === #
    batch_size = zconf.attr(default=8, type=int)
    save_acts = zconf.attr(action="store_true")

    # Specialized config
    no_cuda = zconf.attr(action='store_true')
    fp16 = zconf.attr(action='store_true')
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default='', type=str)
    server_port = zconf.attr(default='', type=str)
    seed = zconf.attr(type=int, default=-1)
    force_overwrite = zconf.attr(action="store_true")


def main(args):
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    with quick_init_out.log_writer.log_context():
        task = tasks.create_task_from_config_path(
            config_path=args.task_config_path,
            verbose=True,
        )
        model_class_spec = model_resolution.resolve_model_setup_classes(
            model_type=args.model_type,
            task_type=task.TASK_TYPE,
        )
        model_wrapper = model_setup.simple_model_setup(
            model_type=args.model_type,
            model_class_spec=model_class_spec,
            config_path=args.model_config_path,
            tokenizer_path=args.model_tokenizer_path,
            task=task,
        )
        model_wrapper.model.roberta.encoder.output_hidden_states = True
        data_obj = DataObj.from_path(
            task=task,
            task_cache_path=args.task_cache_path,
            indices_path=args.indices_path,
            batch_size=args.batch_size,
        )

        # === Compute === #
        act_a = compute_activations_from_path(
            data_obj=data_obj,
            model_wrapper=model_wrapper,
            model_path=args.model_a_path,
            device=quick_init_out.device,
        )
        act_b = compute_activations_from_path(
            data_obj=data_obj,
            model_wrapper=model_wrapper,
            model_path=args.model_b_path,
            device=quick_init_out.device,
        )
        cka_outputs = compute_cka(
            act_a=act_a,
            act_b=act_b,
            device=quick_init_out.device,
        )
        torch.save(cka_outputs, os.path.join(args.output_dir, "cka.p"))
        if args.save_acts:
            torch.save(act_a, os.path.join(args.output_dir, "act_a.p"))
            torch.save(act_b, os.path.join(args.output_dir, "act_b.p"))


@dataclass
class DataObj:
    dataloader: DataLoader
    grouped_example_indices: Sequence
    grouped_position_indices: Sequence

    @classmethod
    def from_path(cls, task, task_cache_path, indices_path, batch_size):
        loaded = torch.load(indices_path)
        grouped_example_indices = loaded["grouped_example_indices"]
        grouped_position_indices = loaded["grouped_position_indices"]
        task_cache = caching.ChunkedFilesDataCache(task_cache_path)
        dataloader = shared_runner.get_eval_dataloader_from_cache(
            eval_cache=task_cache,
            task=task,
            eval_batch_size=batch_size,
            explicit_subset=grouped_example_indices,
        )
        return cls(
            dataloader=dataloader,
            grouped_example_indices=grouped_example_indices,
            grouped_position_indices=grouped_position_indices,
        )


def compute_activations_from_path(data_obj: DataObj,
                                  model_wrapper: model_setup.ModelWrapper,
                                  model_path: str,
                                  device):
    load_model(
        model_wrapper=model_wrapper,
        model_path=model_path,
        device=device,
    )
    return compute_activations_from_model(
        data_obj=data_obj,
        model_wrapper=model_wrapper,
        device=device,
    )


def compute_cka(act_a, act_b, device):
    assert act_a.shape[1] == act_b.shape[1]
    num_layers = act_a.shape[1]
    collated = np.empty([num_layers, num_layers])
    for i in tqdm.tqdm(range(num_layers)):
        for j in tqdm.tqdm(range(num_layers)):
            collated[i, j] = cka.linear_CKA(
                torch.Tensor(act_a[:, i].copy()).float().to(device),
                torch.Tensor(act_b[:, j].copy()).float().to(device),
            ).item()
    return collated


def compute_activations_from_model(data_obj: DataObj,
                                   model_wrapper: model_setup.ModelWrapper,
                                   device):
    collected_acts = []
    with torch.no_grad():
        model_wrapper.model.eval()
        example_i = 0
        for batch, batch_metadata in tqdm.tqdm(data_obj.dataloader):
            batch = batch.to(device)
            outputs = shared_models.forward_batch_basic(
                model=model_wrapper.model,
                batch=batch,
                omit_label_id=True,
            )
            batch_example_indices = torch.LongTensor(np.repeat(
                np.arange(len(batch)),
                [len(data_obj.grouped_position_indices[example_i + i]) for i in range(len(batch))],
            )).to(device)
            batch_position_indices = torch.LongTensor(np.concatenate([
                data_obj.grouped_position_indices[example_i + i] for i in range(len(batch))]
            )).to(device)
            hidden_act = torch.stack(outputs[1], dim=2)
            collected_acts.append(hidden_act[batch_example_indices, batch_position_indices].cpu().numpy())
            example_i += len(batch)
    return np.concatenate(collected_acts)


def load_model(model_wrapper, model_path, device):
    model_setup.simple_load_model_path(
        model=model_wrapper.model,
        model_load_mode="safe",
        model_path=model_path,
        verbose=False,
    )
    model_wrapper.model.to(device)
    return model_wrapper


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
