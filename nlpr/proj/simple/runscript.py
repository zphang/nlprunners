import zconf

import nlpr.shared.initialization as initialization
import nlpr.shared.distributed as distributed
import nlpr.shared.model_setup as model_setup
import nlpr.shared.train_setup as train_setup
import nlpr.tasks as tasks


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    data_dir = zconf.attr(type=str, require=True)
    task_name = zconf.attr(type=str, required=True)
    output_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_load_mode = zconf.attr(type=str, required=True)
    model_save_mode = zconf.attr(default="all", type=str)
    do_lower_case = zconf.attr(action='store_true',
                               help="Set this flag if you are using an uncased model.")
    max_seq_length = zconf.attr(default=128, type=int)

    # === Running Setup === #
    # cache_dir
    do_train = zconf.attr(action='store_true')
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

    # === Training Learning Parameters === #
    learning_rate = zconf.attr(default=1e-5, type=float)
    num_train_epochs = zconf.attr(default=3, type=int)
    max_steps = zconf.attr(default=-1, type=int)  ## Change to None
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
    warmup_steps = zconf.attr(default=0, type=int)   ## warmup proportion?

    # Specialized config
    gradient_accumulation_steps = zconf.attr(default=1, type=int)
    no_cuda = zconf.attr(action='store_true')
    fp16 = zconf.attr(action='store_true')
    fp16_opt_level = zconf.attr(default='O1', type=str)
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default='', type=str)
    server_port = zconf.attr(default='', type=str)


def main(args):
    device, n_gpu = initialization.quick_init(args=args, verbose=True)
    task = tasks.get_task(task_name=args.task_name, data_dir=args.data_dir)

    with distributed.only_first_process(local_rank=args.local_rank):
        # load the model
        model_wrapper = model_setup.ModelWrapper(None, None)  ## todo

    train_examples = task.get_train_examples()
    train_schedule = train_setup.get_train_schedule(
        num_train_examples=len(train_examples),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_gpu_train_batch_size=args.per_gpu_train_batch_size,
        n_gpu=n_gpu,
    )
    optimizer_scheduler = model_setup.create_optimizer(
        model=model_wrapper.model,
        learning_rate=args.learning_rate,
        t_total=args.t_total,
        warmup_steps=args.warmup_steps,
    )
    model_setup.special_model_setup(
        model_wrapper=model_wrapper,
        optimizer_scheduler=optimizer_scheduler,
        fp16=args.fp16, fp16_opt_level=args.fp16_opt_level,
        n_gpu=n_gpu, local_rank=args.local_rank,
    )


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli())
