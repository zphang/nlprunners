import os
import torch

import zconf

import nlpr.shared.initialization as initialization
import nlpr.shared.distributed as distributed
import nlpr.shared.model_setup as model_setup
import nlpr.proj.jiant.modeling.model_setup as jiant_model_setup
import nlpr.proj.jiant.runner as jiant_runner
import nlpr.proj.jiant.components.task_setup as jiant_task_setup
import nlpr.proj.jiant.metarunner as jiant_metarunner
import nlpr.proj.jiant.components.evaluate as jiant_evaluate


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_config_path_dict_path = zconf.attr(type=str, required=True)
    task_cache_config_dict_path = zconf.attr(type=str, required=True)
    sampler_config_path = zconf.attr(type=str, required=True)
    global_train_config_path = zconf.attr(type=str, required=True)
    task_specific_configs_dict_path = zconf.attr(type=str, required=True)
    metric_aggregator_config_path = zconf.attr(type=str, required=True)
    output_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_tokenizer_path = zconf.attr(default=None, type=str)
    model_load_mode = zconf.attr(default="from_ptt", type=str)
    model_save_mode = zconf.attr(default="all", type=str)

    # === Running Setup === #
    do_train = zconf.attr(action='store_true')
    do_val = zconf.attr(action='store_true')
    do_save = zconf.attr(action="store_true")
    no_write_preds = zconf.attr(action="store_true")
    eval_every_steps = zconf.attr(type=int, default=0)
    save_every_steps = zconf.attr(type=int, default=0)
    force_overwrite = zconf.attr(action="store_true")
    seed = zconf.attr(type=int, default=-1)

    # === Training Learning Parameters === #
    learning_rate = zconf.attr(default=1e-5, type=float)
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
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
        jiant_task_container = jiant_task_setup.create_jiant_task_container_from_paths(
            task_config_path_dict_path=args.task_config_path_dict_path,
            task_cache_config_dict_path=args.task_cache_config_dict_path,
            sampler_config_path=args.sampler_config_path,
            global_train_config_path=args.global_train_config_path,
            task_specific_configs_dict_path=args.task_specific_configs_dict_path,
            metric_aggregator_config_path=args.metric_aggregator_config_path,
        )
        with distributed.only_first_process(local_rank=args.local_rank):
            # load the model
            jiant_model = jiant_model_setup.setup_jiant_style_model(
                model_type=args.model_type,
                model_config_path=args.model_config_path,
                tokenizer_path=args.model_tokenizer_path,
                task_dict=jiant_task_container.task_dict,
            )
            jiant_model_setup.delegate_load_from_path(
                jiant_model=jiant_model,
                weights_path=args.model_path,
                load_mode=args.model_load_mode
            )
            jiant_model.to(quick_init_out.device)

        optimizer_scheduler = model_setup.create_optimizer(
            model=jiant_model,
            learning_rate=args.learning_rate,
            t_total=jiant_task_container.global_train_config.max_steps,
            warmup_steps=jiant_task_container.global_train_config.warmup_steps,
            warmup_proportion=None,
            optimizer_type=args.optimizer_type,
            verbose=True,
        )
        jiant_model, optimizer = model_setup.raw_special_model_setup(
            model=jiant_model,
            optimizer=optimizer_scheduler.optimizer,
            fp16=args.fp16, fp16_opt_level=args.fp16_opt_level,
            n_gpu=quick_init_out.n_gpu, local_rank=args.local_rank,
        )
        optimizer_scheduler.optimizer = optimizer
        rparams = jiant_runner.RunnerParameters(
            local_rank=args.local_rank,
            n_gpu=quick_init_out.n_gpu,
            fp16=args.fp16,
            max_grad_norm=args.max_grad_norm,
        )
        runner = jiant_runner.JiantRunner(
            jiant_task_container=jiant_task_container,
            jiant_model=jiant_model,
            optimizer_scheduler=optimizer_scheduler,
            device=quick_init_out.device,
            rparams=rparams,
            log_writer=quick_init_out.log_writer,
        )

        if args.do_train:
            jiant_metarunner.JiantMetarunner(
                runner=runner,
                should_save_func=jiant_metarunner.get_should_save_func(args.save_every_steps),
                should_eval_func=jiant_metarunner.get_should_eval_func(args.eval_every_steps),
                output_dir=args.output_dir,
                verbose=True,
                save_best_model=args.do_save,
                load_best_model=True,
                log_writer=quick_init_out.log_writer,
            ).run_train_loop()

        if args.do_save:
            torch.save(
                jiant_model.state_dict(),
                os.path.join(args.output_dir, "model.p")
            )

        if args.do_val:
            val_results_dict = runner.run_val()
            jiant_evaluate.write_val_results(
                val_results_dict=val_results_dict,
                metrics_aggregator=jiant_task_container.metrics_aggregator,
                output_dir=args.output_dir,
                verbose=True,
            )


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
