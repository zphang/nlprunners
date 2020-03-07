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
import nlpr.proj.weight_study.split_dict as split_dict
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
    skip_b = zconf.attr(action="store_true")
    skip_cka = zconf.attr(action="store_true")
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
            task=task,
            model_wrapper=model_wrapper,
            model_path=args.model_a_path,
            device=quick_init_out.device,
        )
        if not args.skip_b:
            act_b = compute_activations_from_path(
                data_obj=data_obj,
                task=task,
                model_wrapper=model_wrapper,
                model_path=args.model_b_path,
                device=quick_init_out.device,
            )
        if not args.skip_cka:
            assert not args.skip_b
            cka_outputs = compute_cka(
                act_a=act_a,
                act_b=act_b,
                device=quick_init_out.device,
            )
            torch.save(cka_outputs, os.path.join(args.output_dir, "cka.p"))
        if args.save_acts:
            torch.save(act_a, os.path.join(args.output_dir, "act_a.p"))
            if not args.skip_b:
                torch.save(act_b, os.path.join(args.output_dir, "act_b.p"))


@dataclass
class DataObj:
    dataloader: DataLoader
    grouped_input_indices: Sequence
    grouped_position_indices: Sequence

    @classmethod
    def from_path(cls, task, task_cache_path, indices_path, batch_size):
        loaded = torch.load(indices_path)
        grouped_input_indices = loaded["grouped_input_indices"]
        grouped_position_indices = loaded["grouped_position_indices"]
        task_cache = caching.ChunkedFilesDataCache(task_cache_path)
        dataloader = shared_runner.get_eval_dataloader_from_cache(
            eval_cache=task_cache,
            task=task,
            eval_batch_size=batch_size,
            explicit_subset=np.array(grouped_input_indices) // get_num_inputs(task),
        )
        return cls(
            dataloader=dataloader,
            grouped_input_indices=grouped_input_indices,
            grouped_position_indices=grouped_position_indices,
        )


def get_num_inputs(task: tasks.Task):
    if task.TASK_TYPE == tasks.TaskTypes.MULTIPLE_CHOICE:
        return task.NUM_CHOICES
    else:
        return 1


def compute_activations_from_path(data_obj: DataObj,
                                  task: tasks.Task,
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
        task=task,
        model_wrapper=model_wrapper,
        device=device,
    )


def compute_cka(act_a, act_b, device):
    assert act_a.shape[1] == act_b.shape[1]
    num_layers = act_a.shape[1]
    collated = np.empty([num_layers, num_layers])
    for i in tqdm.tqdm(range(num_layers), desc="CKA row"):
        act_a_tensor = torch.Tensor(act_a[:, i].copy()).float().to(device)
        for j in tqdm.tqdm(range(num_layers)):
            act_b_tensor = torch.Tensor(act_b[:, j].copy()).float().to(device)
            collated[i, j] = cka.linear_CKA(act_a_tensor, act_b_tensor).item()
    return collated


def get_hidden_act(task: tasks.Task, model_wrapper, batch):
    if task.TASK_TYPE in (
                tasks.TaskTypes.CLASSIFICATION,
                tasks.TaskTypes.MULTIPLE_CHOICE,
                tasks.TaskTypes.REGRESSION,
            ):
        _, raw_hidden = shared_models.forward_batch_basic(
            model=model_wrapper.model,
            batch=batch,
            omit_label_id=True,
        )
    elif task.TASK_TYPE in (
                tasks.TaskTypes.SPAN_COMPARISON_CLASSIFICATION,
            ):
        _, raw_hidden = model_wrapper.model(
            input_ids=batch.input_ids,
            spans=batch.spans,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            labels=None,
        )
    elif task.TASK_TYPE == tasks.TaskTypes.SQUAD_STYLE_QA:
        start_positions = None
        end_positions = None
        # TODO: Refactor this wrt model_resolution
        # Actually "xlm", "roberta", "distilbert"
        if model_wrapper.model.__class__.__name__.startswith("Robert"):
            segment_ids = None
        else:
            segment_ids = batch.segment_ids
        _, _, raw_hidden = model_wrapper.model(
            input_ids=batch.input_ids,
            attention_mask=batch.input_mask,
            token_type_ids=segment_ids,
            start_positions=start_positions,
            end_positions=end_positions,
        )
    elif task.TASK_TYPE == tasks.TaskTypes.TAGGING:
        _, raw_hidden = model_wrapper.model(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
        )
    else:
        raise KeyError(task.TASK_TYPE)
    hidden_act = torch.stack(raw_hidden, dim=2)
    return hidden_act


def compute_activations_from_model(data_obj: DataObj,
                                   task: tasks.Task,
                                   model_wrapper: model_setup.ModelWrapper,
                                   device):
    num_inputs_per_example = get_num_inputs(task)
    collected_acts = []
    with torch.no_grad():
        model_wrapper.model.eval()
        example_i = 0
        for batch, batch_metadata in tqdm.tqdm(data_obj.dataloader, desc="Computing Activation"):
            batch = batch.to(device)

            hidden_act = get_hidden_act(
                task=task,
                model_wrapper=model_wrapper,
                batch=batch,
            )

            if task.TASK_TYPE == tasks.TaskTypes.MULTIPLE_CHOICE:
                input_indices = np.repeat(
                    np.arange(len(batch)),
                    [len(data_obj.grouped_position_indices[example_i + i])
                     for i in range(len(batch))],
                )
                batch_example_indices = torch.LongTensor(input_indices // num_inputs_per_example).to(device)
                batch_choice_indices = torch.LongTensor(input_indices % num_inputs_per_example).to(device)
                batch_position_indices = torch.LongTensor(np.concatenate([
                    data_obj.grouped_position_indices[example_i + i] for i in range(len(batch))]
                )).to(device)
                collected_acts.append(hidden_act[
                    batch_example_indices, batch_choice_indices, :, batch_position_indices
                ].cpu().numpy())
                example_i += len(batch)
            else:
                batch_example_indices = torch.LongTensor(np.repeat(
                    np.arange(len(batch)),
                    [len(data_obj.grouped_position_indices[example_i + i])
                     for i in range(len(batch))],
                )).to(device)
                batch_position_indices = torch.LongTensor(np.concatenate([
                    data_obj.grouped_position_indices[example_i + i] for i in range(len(batch))]
                )).to(device)
                collected_acts.append(hidden_act[batch_example_indices, batch_position_indices].cpu().numpy())
                example_i += len(batch)
    return np.concatenate(collected_acts)


def load_model(model_wrapper, model_path, device):
    if model_path.endswith("split_dict"):
        state_dict = split_dict.load_split_dict(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")
    model_setup.simple_load_model(
        model=model_wrapper.model,
        model_load_mode="safe",
        state_dict=state_dict,
        verbose=False,
    )
    model_wrapper.model.to(device)
    return model_wrapper


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
