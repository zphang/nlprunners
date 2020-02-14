import os
import numpy as np

import zconf
import pyutils.io as io

import nlpr.tasks as tasks
import nlpr.shared.runner as shared_runner
import nlpr.shared.model_resolution as shared_model_resolution
import nlpr.shared.model_setup as shared_model_setup
import nlpr.shared.caching as shared_caching
import nlpr.tasks.evaluate as evaluate
import nlpr.shared.torch_utils as torch_utils


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_config_path = zconf.attr(type=str, required=True)
    model_type = zconf.attr(type=str, required=True)
    model_tokenizer_path = zconf.attr(type=str, required=True)
    output_dir = zconf.attr(type=str, required=True)

    # === Optional parameters === #
    phases = zconf.attr(default="train,val", type=str)
    max_seq_length = zconf.attr(default=128, type=int)
    chunk_size = zconf.attr(default=10000, type=int)
    force_overwrite = zconf.attr(action="store_true")
    smart_truncate = zconf.attr(action="store_true")


def chunk_and_save(phase, examples, feat_spec, tokenizer, args: RunConfiguration):
    dataset = shared_runner.convert_examples_to_dataset(
        examples=examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
        phase=phase,
        verbose=True,
    )
    if args.smart_truncate:
        dataset, length = experimental_smart_truncate(
            dataset=dataset,
            max_seq_length=args.max_seq_length,
        )
        os.makedirs(os.path.join(args.output_dir, phase), exist_ok=True)
        io.write_json(
            data={"truncated_to": int(length)},
            path=os.path.join(args.output_dir, phase, "smart_truncate.json"),
        )
    shared_caching.chunk_and_save(
        data=dataset.data,
        chunk_size=args.chunk_size,
        data_args=args.to_dict(),
        output_dir=os.path.join(args.output_dir, phase),
    )


def experimental_smart_truncate(dataset: torch_utils.ListDataset,
                                max_seq_length: int):
    if "input_mask" not in dataset.data[0]["data_row"].get_fields():
        raise RuntimeError("Smart truncate not supported")
    max_valid_length_ls = []
    range_idx = np.arange(max_seq_length)
    for datum in dataset.data:
        indexer = datum["data_row"].input_mask.reshape(-1, max_seq_length).max(-2)
        max_valid_length_ls.append(range_idx[indexer.astype(bool)].max() + 1)
    max_valid_length = max(max_valid_length_ls)

    if max_valid_length == max_seq_length:
        return dataset, max_seq_length

    new_datum_ls = []
    for datum in dataset.data:
        row_dict = datum["data_row"].asdict()
        new_row_dict = row_dict.copy()
        for k, v in row_dict.items():
            if not isinstance(v, np.ndarray):
                continue
            if max_seq_length not in v.shape:
                continue
            if not v.shape.count(max_seq_length) == 1:
                raise RuntimeError("confusing dimensions")
            slice_ls = []
            for n in v.shape:
                if n == max_seq_length:
                    slice_ls.append(slice(None, max_valid_length))
                else:
                    slice_ls.append(slice(None))
            new_row_dict[k] = v[tuple(slice_ls)]
        new_datum_ls.append({
            "data_row": datum["data_row"].__class__(**new_row_dict),
            "metadata": datum["metadata"],
        })
    new_dataset = torch_utils.ListDataset(new_datum_ls)
    return new_dataset, max_valid_length


def main(args: RunConfiguration):
    task = tasks.create_task_from_config_path(
        config_path=args.task_config_path,
        verbose=True,
    )
    feat_spec = shared_model_resolution.build_featurization_spec(
        model_type=args.model_type,
        max_seq_length=args.max_seq_length,
    )
    model_class_spec = shared_model_resolution.resolve_model_setup_classes(
        model_type=args.model_type,
        task_type=task.TASK_TYPE,
    )
    tokenizer = shared_model_setup.get_tokenizer(
        model_type=args.model_type,
        tokenizer_class=model_class_spec.tokenizer_class,
        tokenizer_path=args.model_tokenizer_path,
    )
    phases = args.phases.split(",")
    assert set(phases) < {"train", "val", "test"}
    if "train" in phases:
        chunk_and_save(
            phase="train",
            examples=task.get_train_examples(),
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )
    if "val" in phases:
        val_examples = task.get_val_examples()
        chunk_and_save(
            phase="val",
            examples=val_examples,
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )
        shared_caching.chunk_and_save(
            data=list(evaluate.get_labels_from_examples(task=task, examples=val_examples)),
            chunk_size=args.chunk_size,
            data_args=args.to_dict(),
            output_dir=os.path.join(args.output_dir, "val_labels"),
        )
    if "test" in phases:
        chunk_and_save(
            phase="test",
            examples=task.get_test_examples(),
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
