import os

import zconf
import pyutils.io as io

import nlpr.tasks as tasks
import nlpr.shared.model_resolution as shared_model_resolution
import nlpr.shared.model_setup as shared_model_setup
import nlpr.shared.caching as shared_caching
import nlpr.tasks.evaluate as evaluate
from nlpr.constants import PHASE
import nlpr.shared.preprocessing as preprocessing


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
    do_iter = zconf.attr(action="store_true")


def chunk_and_save(phase, examples, feat_spec, tokenizer, args: RunConfiguration):
    if args.do_iter:
        iter_chunk_and_save(
            phase=phase,
            examples=examples,
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args
        )
    else:
        full_chunk_and_save(
            phase=phase,
            examples=examples,
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args
        )


def full_chunk_and_save(phase, examples, feat_spec, tokenizer, args: RunConfiguration):
    dataset = preprocessing.convert_examples_to_dataset(
        examples=examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
        phase=phase,
        verbose=True,
    )
    if args.smart_truncate:
        dataset, length = preprocessing.experimental_smart_truncate(
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


def iter_chunk_and_save(phase, examples, feat_spec, tokenizer, args: RunConfiguration):
    dataset_generator = preprocessing.iter_chunk_convert_examples_to_dataset(
        examples=examples,
        feat_spec=feat_spec,
        tokenizer=tokenizer,
        phase=phase,
        verbose=True,
    )
    max_valid_length_recorder = preprocessing.MaxValidLengthRecorder(args.max_seq_length)
    shared_caching.iter_chunk_and_save(
        data=dataset_generator,
        chunk_size=args.chunk_size,
        data_args=args.to_dict(),
        output_dir=os.path.join(args.output_dir, phase),
        recorder_callback=max_valid_length_recorder,
    )
    if args.smart_truncate:
        preprocessing.experimental_smart_truncate_cache(
            cache=shared_caching.ChunkedFilesDataCache(os.path.join(args.output_dir, phase)),
            max_seq_length=args.max_seq_length,
            max_valid_length=max_valid_length_recorder.max_valid_length,
        )
        io.write_json(
            data={"truncated_to": int(max_valid_length_recorder.max_valid_length)},
            path=os.path.join(args.output_dir, phase, "smart_truncate.json"),
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
    assert set(phases) < {PHASE.TRAIN, PHASE.VAL, PHASE.TEST}
    if PHASE.TRAIN in phases:
        chunk_and_save(
            phase=PHASE.TRAIN,
            examples=task.get_train_examples(),
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )
    if PHASE.VAL in phases:
        val_examples = task.get_val_examples()
        chunk_and_save(
            phase=PHASE.VAL,
            examples=val_examples,
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )
        shared_caching.chunk_and_save(
            data=evaluate.get_labels_from_cache(
                task=task,
                cache=shared_caching.ChunkedFilesDataCache(os.path.join(args.output_dir, PHASE.VAL)),
            ),
            chunk_size=args.chunk_size,
            data_args=args.to_dict(),
            output_dir=os.path.join(args.output_dir, "val_labels"),
        )
    if PHASE.TEST in phases:
        chunk_and_save(
            phase=PHASE.TEST,
            examples=task.get_test_examples(),
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
