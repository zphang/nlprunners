import os
import torch

import zconf

import nlpr.shared.initialization as initialization
import nlpr.shared.distributed as distributed
import nlpr.shared.model_setup as model_setup
import nlpr.shared.model_resolution as model_resolution
import nlpr.shared.train_setup as train_setup
import nlpr.tasks as tasks
import nlpr.tasks.evaluate as evaluate
import nlpr.proj.simple.runner as simple_runner
import nlpr.shared.metarunner as metarunner
import nlpr.shared.caching as caching


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_config_path = zconf.attr(type=str, required=True)
    task_cache_data_path = zconf.attr(type=str, required=True)
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
    do_train = zconf.attr(action='store_true')
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
    train_examples_number = zconf.attr(type=int, default=None)
    train_examples_fraction = zconf.attr(type=float, default=None)

    # === Training Learning Parameters === #
    learning_rate = zconf.attr(default=1e-5, type=float)
    num_train_epochs = zconf.attr(default=3, type=int)
    max_steps = zconf.attr(default=-1, type=int)  ## Change to None
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
    warmup_steps = zconf.attr(default=None, type=int)
    warmup_proportion = zconf.attr(default=0.1, type=float)
    optimizer_type = zconf.attr(default="adam", type=str)

    # Specialized config
    gradient_accumulation_steps = zconf.attr(default=1, type=int)
    no_cuda = zconf.attr(action='store_true')
    fp16 = zconf.attr(action='store_true')
    fp16_opt_level = zconf.attr(default='O1', type=str)
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default='', type=str)
    server_port = zconf.attr(default='', type=str)


def main(args):
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    with quick_init_out.log_writer.log_context():
        task = tasks.create_task_from_config_path(
            config_path=args.task_config_path,
            verbose=True,
        )

        with distributed.only_first_process(local_rank=args.local_rank):
            # load the model
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
            model_setup.simple_load_model_path(
                model=model_wrapper.model,
                model_load_mode=args.model_load_mode,
                model_path=args.model_path,
                verbose=True,
            )
            model_wrapper.model.to(quick_init_out.device)

        train_cache = caching.ChunkedFilesDataCache(os.path.join(args.task_cache_data_path, "train"))
        num_train_examples = len(train_cache)

        train_schedule = train_setup.get_train_schedule(
            num_train_examples=num_train_examples,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_gpu_train_batch_size=args.train_batch_size,
            n_gpu=quick_init_out.n_gpu,
        )
        quick_init_out.log_writer.write_entry("text", f"t_total: {train_schedule.t_total}", do_print=True)
        loss_criterion = train_setup.resolve_loss_function(task_type=task.TASK_TYPE)
        optimizer_scheduler = model_setup.create_optimizer(
            model=model_wrapper.model,
            learning_rate=args.learning_rate,
            t_total=train_schedule.t_total,
            warmup_steps=args.warmup_steps,
            warmup_proportion=args.warmup_proportion,
            optimizer_type=args.optimizer_type,
            verbose=True,
        )
        model_setup.special_model_setup(
            model_wrapper=model_wrapper,
            optimizer_scheduler=optimizer_scheduler,
            fp16=args.fp16, fp16_opt_level=args.fp16_opt_level,
            n_gpu=quick_init_out.n_gpu, local_rank=args.local_rank,
        )
        rparams = simple_runner.RunnerParameters(
            feat_spec=model_resolution.build_featurization_spec(
                model_type=args.model_type,
                max_seq_length=args.max_seq_length,
            ),
            local_rank=args.local_rank,
            n_gpu=quick_init_out.n_gpu,
            fp16=args.fp16,
            learning_rate=args.learning_rate,
            eval_batch_size=args.eval_batch_size,
            max_grad_norm=args.max_grad_norm,
        )
        runner = simple_runner.SimpleTaskRunner(
            task=task,
            model_wrapper=model_wrapper,
            optimizer_scheduler=optimizer_scheduler,
            loss_criterion=loss_criterion,
            device=quick_init_out.device,
            rparams=rparams,
            train_schedule=train_schedule,
            log_writer=quick_init_out.log_writer,
        )

        if args.do_train:
            val_cache = caching.ChunkedFilesDataCache(os.path.join(args.task_cache_data_path, "val"))
            metarunner.MetaRunner(
                runner=runner,
                train_cache=train_cache,
                val_cache=val_cache,
                partial_eval_number=args.partial_eval_number,
                should_save_func=metarunner.get_should_save_func(args.save_every_steps),
                should_eval_func=metarunner.get_should_eval_func(args.eval_every_steps),
                output_dir=args.output_dir,
                verbose=True,
                save_best_model=args.do_save,
                load_best_model=True,
                log_writer=quick_init_out.log_writer,
            ).train_val_save_every()

        if args.do_save:
            torch.save(
                model_wrapper.model.state_dict(),
                os.path.join(args.output_dir, "model.p")
            )

        if args.do_val:
            val_cache = caching.ChunkedFilesDataCache(os.path.join(args.task_cache_data_path, "val"))
            results = runner.run_val(val_cache)
            evaluate.write_val_results(
                results=results,
                output_dir=args.output_dir,
                verbose=True,
            )

        if args.do_test:
            test_examples = task.get_test_examples()
            logits = runner.run_test(test_examples)
            evaluate.write_preds(
                logits=logits,
                output_path=os.path.join(args.output_dir, "test_preds.csv"),
            )


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
