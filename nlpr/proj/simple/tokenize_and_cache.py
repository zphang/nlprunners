import os

import zconf

import nlpr.tasks as tasks
import nlpr.shared.runner as shared_runner
import nlpr.shared.model_resolution as shared_model_resolution
import nlpr.shared.model_setup as shared_model_setup
import nlpr.shared.caching as shared_caching


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


def chunk_and_save(phase, examples, feat_spec, tokenizer, task, args):
    dataset_with_metadata = shared_runner.convert_examples_to_dataset(
        examples=examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
        task=task,
        verbose=True,
    )
    shared_caching.chunk_and_save(
        dataset_with_metadata=dataset_with_metadata,
        chunk_size=args.chunk_size,
        data_args=args.to_dict(),
        output_dir=os.path.join(args.output_dir, phase),
    )


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
            task=task,
            args=args,
        )
    if "val" in phases:
        chunk_and_save(
            phase="val",
            examples=task.get_val_examples(),
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            task=task,
            args=args,
        )
    if "test" in phases:
        chunk_and_save(
            phase="test",
            examples=task.get_test_examples(),
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            task=task,
            args=args,
        )


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
