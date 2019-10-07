import os
import random
import torch

import zconf

import nlpr.shared.initialization as initialization

import nlpr.tasks as tasks
import nlpr.tasks.evaluate as evaluate
import nlpr.proj.n_training.runner as n_training_runner
import nlpr.shared.unsup.load_data as unsup_load_data


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_config_path = zconf.attr(type=str, required=True)
    unsup_task_config_path = zconf.attr(type=str, required=True)
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
    eval_every_steps = zconf.attr(type=int, default=0)
    save_every_steps = zconf.attr(type=int, default=0)
    partial_eval_number = zconf.attr(type=int, default=1000)
    train_batch_size = zconf.attr(default=8, type=int)  # per gpu
    eval_batch_size = zconf.attr(default=8, type=int)  # per gpu
    force_overwrite = zconf.attr(action="store_true")
    # overwrite_cache = zconf.attr(action="store_true")
    seed = zconf.attr(type=int, default=-1)

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
    with_disagreement = zconf.attr(action='store_true')
    confidence_threshold = zconf.attr(default=None, type=float)
    num_unlabeled = zconf.attr(default=-1, type=int)


def main(args):
    if os.path.exists(os.path.join(args.output_dir, "val_metrics.json")):
        print("HACK TO SKIP JOBS")
        return
    quick_init_out = initialization.quick_init(args=args, verbose=True)

    with quick_init_out.log_writer.log_context():
        task = tasks.create_task_from_config_path(
            config_path=args.task_config_path,
            verbose=True,
        )
        unsup_task, unsup_data = \
            unsup_load_data.load_unsup_examples_from_config_path(args.unsup_task_config_path)

        labeled_train_examples = task.get_train_examples()
        unlabeled_train_examples = unsup_data["orig"]
        if args.num_unlabeled != -1:
            random.shuffle(unlabeled_train_examples)
            unlabeled_train_examples = unlabeled_train_examples[:args.num_unlabeled]

        n_training_rparams = n_training_runner.NTrainingRunnerParameters(
            num_models=args.num_models,
            num_iter=args.num_iter,
            with_disagreement=args.with_disagreement,
            confidence_threshold=args.confidence_threshold,
        )

        runner_creator = n_training_runner.RunnerCreator(
            task=task,
            model_type=args.model_type,  model_path=args.model_path,
            model_config_path=args.model_config_path, model_tokenizer_path=args.model_tokenizer_path,
            model_load_mode=args.model_load_mode,
            learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
            warmup_proportion=args.warmup_proportion,
            train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps, gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm, max_seq_length=args.max_seq_length,
            local_rank=args.local_rank, fp16=args.fp16, fp16_opt_level=args.fp16_opt_level,
            device=quick_init_out.device, n_gpu=quick_init_out.n_gpu,
            verbose=False,
        )
        meta_runner_parameters = n_training_runner.MetaRunnerParameters(
            partial_eval_number=args.partial_eval_number,
            save_every_steps=args.save_every_steps,
            eval_every_steps=args.eval_every_steps,
            output_dir=args.output_dir,
            do_save=args.do_save,
        )
        runner = n_training_runner.NTrainingRunner(
            runner_creator=runner_creator,
            labeled_examples=labeled_train_examples,
            unlabeled_examples=unlabeled_train_examples,
            rparams=n_training_rparams,
            log_writer=quick_init_out.log_writer,
            meta_runner_parameters=meta_runner_parameters,
        )
        sub_runner_ls = runner.train()

        if args.do_save:
            for i, sub_runner in enumerate(sub_runner_ls):
                torch.save(
                    sub_runner.model_wrapper.model.state_dict(),
                    os.path.join(args.output_dir, f"model__{i}.p")
                )

        if args.do_val:
            val_examples = task.get_val_examples()
            logits_ls = []
            for i, sub_runner in enumerate(sub_runner_ls):
                n_training_runner.runner_reactivate(sub_runner)
                results = sub_runner.run_val(val_examples)
                n_training_runner.runner_save_memory(sub_runner)
                sub_output_dir = os.path.join(args.output_dir)
                evaluate.write_val_results(
                    results=results,
                    output_dir=sub_output_dir,
                    verbose=True,
                )
                logits_ls.append(results["logits"])
            print("N_Training")
            n_training_runner.write_n_training_val_results(
                task=task,
                logits_ls=logits_ls,
                val_examples=val_examples,
                output_dir=args.output_dir,
                verbose=True,
            )

        if args.do_test:
            raise NotImplementedError()

    initialization.write_done(args.output_dir)


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
