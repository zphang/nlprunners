import os
import torch

import zconf

import nlpr.shared.initialization as initialization

import nlpr.tasks as tasks
import nlpr.tasks.evaluate as evaluate
import nlpr.proj.uda.load_data as load_data
import nlpr.proj.n_training.runner as n_training_runner


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_config_path = zconf.attr(type=str, required=True)
    uda_task_config_path = zconf.attr(type=str, required=True)
    output_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_tokenizer_path = zconf.attr(default=None, type=str)
    model_load_mode = zconf.attr(default="safe", type=str)
    model_save_mode = zconf.attr(default="all", type=str)
    max_seq_length = zconf.attr(default=128, type=int)

    # === Running Setup === #
    # cache_dir
    do_val = zconf.attr(action='store_true')
    do_test = zconf.attr(action='store_true')
    do_save = zconf.attr(action="store_true")
    do_val_history = zconf.attr(action='store_true')
    train_save_every = zconf.attr(type=int, default=None)
    train_save_every_epoch = zconf.attr(action="store_true")
    eval_every_epoch = zconf.attr(action="store_true")
    eval_every = zconf.attr(type=int, default=None)
    train_batch_size = zconf.attr(default=8, type=int)  # per gpu
    eval_batch_size = zconf.attr(default=8, type=int)  # per gpu
    force_overwrite = zconf.attr(action="store_true")
    # overwrite_cache = zconf.attr(action="store_true")
    seed = zconf.attr(type=int, default=-1)
    use_tensorboard = zconf.attr(action="store_true")

    # === Training Learning Parameters === #
    learning_rate = zconf.attr(default=1e-5, type=float)
    num_train_epochs = zconf.attr(default=3, type=int)
    max_steps = zconf.attr(default=-1, type=int)  ## Change to None
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
    warmup_steps = zconf.attr(default=None, type=int)
    warmup_proportion = zconf.attr(default=0.1, type=float)

    # Specialized config
    gradient_accumulation_steps = zconf.attr(default=1, type=int)
    no_cuda = zconf.attr(action='store_true')
    fp16 = zconf.attr(action='store_true')
    fp16_opt_level = zconf.attr(default='O1', type=str)
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default='', type=str)
    server_port = zconf.attr(default='', type=str)

    # N-Training config
    num_models = zconf.attr(default=3, type=int)
    num_iter = zconf.attr(default=10, type=int)


def main(args):
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    task = tasks.create_task_from_config_path(
        config_path=args.task_config_path,
        verbose=True,
    )
    unlabeled_task, unlabeled_task_data = \
        load_data.load_task_data_from_path(args.uda_task_config_path)

    labeled_train_examples = task.get_train_examples()
    unlabeled_train_examples = unlabeled_task_data["unsup"]["orig"]

    n_training_rparams = n_training_runner.NTrainingRunnerParameters(
        num_models=args.num_models,
        num_iter=args.num_iter,
    )

    runner_creator = n_training_runner.RunnerCreator(
        task=task,
        model_type=args.model_type,  model_path=args.model_path,
        model_config_path=args.model_config_path, model_tokenizer_path=args.model_tokenizer_path,
        model_load_mode=args.model_load_mode,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        warmup_proportion=args.warmup_proportion,
        train_batch_size=args.train_batch_size, eval_batch_size=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps, gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm, max_seq_length=args.max_seq_length,
        local_rank=args.local_rank, fp16=args.fp16, fp16_opt_level=args.fp16_opt_level,
        device=quick_init_out.device, n_gpu=quick_init_out.n_gpu,
        verbose=False,
    )
    runner = n_training_runner.NTrainingRunner(
        runner_creator=runner_creator,
        labeled_examples=labeled_train_examples,
        unlabeled_examples=unlabeled_train_examples,
        rparams=n_training_rparams,
        log_writer=quick_init_out.log_writer,
    )

    with quick_init_out.log_writer.log_context():
        sub_runner_ls = runner.train()

        if args.do_save:
            for i, sub_runner in enumerate(sub_runner_ls):
                torch.save(
                    sub_runner.model_wrapper.model.state_dict(),
                    os.path.join(args.output_dir, f"model__{i}.p")
                )

        if args.do_val:
            val_examples = task.get_val_examples()
            for i, sub_runner in enumerate(sub_runner_ls):
                results = sub_runner.run_val(val_examples)
                sub_output_dir = os.path.join(args.output_dir)
                evaluate.write_val_results(
                    results=results,
                    output_dir=sub_output_dir,
                    verbose=True,
                )

        if args.do_test:
            raise NotImplementedError()


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
