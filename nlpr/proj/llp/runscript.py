import os
import torch

import zconf

import nlpr.shared.initialization as initialization
import nlpr.shared.distributed as distributed
import nlpr.shared.model_setup as shared_model_setup
import nlpr.shared.model_resolution as model_resolution
import nlpr.shared.train_setup as train_setup
import nlpr.tasks.evaluate as evaluate
import nlpr.tasks as tasks
import nlpr.shared.metarunner as metarunner

import nlpr.proj.llp.runner as llp_runner
import nlpr.proj.llp.model_setup as llp_model_setup

from pyutils.io import write_json


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_config_path = zconf.attr(type=str, required=True)
    full_task_config_path = zconf.attr(type=str, required=True)
    output_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_tokenizer_path = zconf.attr(default=None, type=str)
    model_load_mode = zconf.attr(type=str, required=True)
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

    # LLP hyperparams
    llp_embedding_dim = zconf.attr(type=int, default=128)
    llp_const_k = zconf.attr(type=int, default=10)
    llp_const_t = zconf.attr(type=int, default=25)
    llp_const_tau = zconf.attr(type=float, default=0.07)
    llp_prop_chunk_size = zconf.attr(type=int, default=500)
    llp_mem_bank_t = zconf.attr(type=float, default=0.5)
    llp_rep_global_agg_loss_lambda = zconf.attr(type=float, default=1.)
    llp_embedding_norm_loss = zconf.attr(type=float, default=0.01)
    llp_compute_global_agg_loss_mode = zconf.attr(type=str, default="v1")
    llp_load_override = zconf.attr(type=str, default=None)

    unlabeled_train_examples_number = zconf.attr(type=int, default=None)
    unlabeled_train_examples_fraction = zconf.attr(type=float, default=None)


def main(args):
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    task = tasks.create_task_from_config_path(config_path=args.task_config_path)

    with distributed.only_first_process(local_rank=args.local_rank):
        # load the model
        model_wrapper = llp_model_setup.setup_model(
            model_type=args.model_type,
            task=task,
            llp_embedding_dim=args.llp_embedding_dim,
            config_path=args.model_config_path,
            tokenizer_path=args.model_tokenizer_path,
        )
        llp_model_setup.load_model_path(
            model=model_wrapper.model,
            model_path=args.model_path,
            model_load_mode=args.model_load_mode,
        )
        model_wrapper.model.to(quick_init_out.device)

    # === Train Data Setup [START] === #
    labeled_examples = task.get_train_examples()
    # VERY hacky
    unlabeled_task = tasks.create_task_from_config_path(args.full_task_config_path)
    unlabeled_examples, indices = train_setup.maybe_subsample_train(
        train_examples=unlabeled_task.get_train_examples(),
        train_examples_number=args.unlabeled_train_examples_number,
        train_examples_fraction=args.unlabeled_train_examples_fraction,
    )
    for example in unlabeled_examples:
        example.label = task.LABELS[-1]
    train_examples = labeled_examples + unlabeled_examples
    if indices is not None:
        write_json(indices, os.path.join(args.output_dir, "sampled_indices.json"))
    num_train_examples = len(train_examples)
    # === Train Data Setup [END] === #

    train_schedule = train_setup.get_train_schedule(
        num_train_examples=num_train_examples,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_gpu_train_batch_size=args.train_batch_size,
        n_gpu=quick_init_out.n_gpu,
    )
    print("t_total", train_schedule.t_total)
    loss_criterion = train_setup.resolve_loss_function(task_type=task.TASK_TYPE)
    optimizer_scheduler = shared_model_setup.create_optimizer(
        model=model_wrapper.model,
        learning_rate=args.learning_rate,
        t_total=train_schedule.t_total,
        warmup_steps=args.warmup_steps,
        warmup_proportion=args.warmup_proportion,
        optimizer_type=args.optimizer_type,
        verbose=True,
    )

    # I don't think this works for LLP...
    shared_model_setup.special_model_setup(
        model_wrapper=model_wrapper,
        optimizer_scheduler=optimizer_scheduler,
        fp16=args.fp16, fp16_opt_level=args.fp16_opt_level,
        n_gpu=quick_init_out.n_gpu, local_rank=args.local_rank,
    )
    rparams = llp_runner.RunnerParameters(
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
    llp_params = llp_runner.LlpParameters(
        num_labeled=len(labeled_examples),
        llp_embedding_dim=args.llp_embedding_dim,
        llp_const_k=args.llp_const_k,
        llp_const_t=args.llp_const_t,
        llp_const_tau=args.llp_const_tau,
        llp_prop_chunk_size=args.llp_prop_chunk_size,
        llp_mem_bank_t=args.llp_mem_bank_t,
        llp_rep_global_agg_loss_lambda=args.llp_rep_global_agg_loss_lambda,
        llp_embedding_norm_loss=args.llp_embedding_norm_loss,
        llp_compute_global_agg_loss_mode=args.llp_compute_global_agg_loss_mode,
    )
    runner = llp_runner.LLPRunner(
        task=task,
        model_wrapper=model_wrapper,
        optimizer_scheduler=optimizer_scheduler,
        loss_criterion=loss_criterion,
        device=quick_init_out.device,
        rparams=rparams,
        llp_params=llp_params,
        train_schedule=train_schedule,
        log_writer=quick_init_out.log_writer,
    )

    with quick_init_out.log_writer.log_context():
        if args.do_train:
            runner.init_llp_state(train_examples)
            # runner.run_train(train_examples)
            val_examples = task.get_val_examples()
            llp_runner.train_val_save_every(
                runner=runner,
                train_examples=train_examples,
                val_examples=val_examples[:args.partial_eval_number],  # quick and dirty
                should_save_func=metarunner.get_should_save_func(args.save_every_steps),
                should_eval_func=metarunner.get_should_eval_func(args.eval_every_steps),
                output_dir=args.output_dir,
                verbose=True,
                save_best_model=args.do_save,
                load_best_model=True,
                log_writer=quick_init_out.log_writer,
            )

        if args.do_save:
            torch.save(
                model_wrapper.model.state_dict(),
                os.path.join(args.output_dir, "model.p")
            )

        if args.do_val:
            val_examples = task.get_val_examples()
            results = runner.run_val(val_examples)
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
