import nlpr.shared.initialization as initialization
import nlpr.proj.jiant.runscript as jiant_runscript

import torch

import zconf
import pyutils.datastructures as datastructures
import pyutils.io as io
import pyutils.display as display


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_config_path_dict_path = zconf.attr(type=str, required=True)
    task_cache_config_dict_path = zconf.attr(type=str, required=True)
    sampler_config_path = zconf.attr(type=str, required=True)
    global_train_config_path = zconf.attr(type=str, required=True)
    task_specific_configs_dict_path = zconf.attr(type=str, required=True)
    metric_aggregator_config_path = zconf.attr(type=str, required=True)
    output_path = zconf.attr(type=str, required=True)

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_tokenizer_path = zconf.attr(default=None, type=str)
    model_load_mode = zconf.attr(default="from_ptt", type=str)
    model_save_mode = zconf.attr(default="all", type=str)

    # === Training Learning Parameters (Unused) === #
    learning_rate = zconf.attr(default=1e-5, type=float)
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
    optimizer_type = zconf.attr(default="adam", type=str)

    # Specialized config (Unused)
    gradient_accumulation_steps = zconf.attr(default=1, type=int)
    no_cuda = zconf.attr(action='store_true')
    fp16 = zconf.attr(action='store_true')
    fp16_opt_level = zconf.attr(default='O1', type=str)
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default='', type=str)
    server_port = zconf.attr(default='', type=str)


def main(args: RunConfiguration):
    quick_init_out = initialization.QuickInitContainer(
        n_gpu=1,
        device=torch.device("cuda:0"),
        log_writer=None,
    )
    runner = jiant_runscript.setup_runner(
        args=args,
        quick_init_out=quick_init_out,
    )
    runner.jiant_model.encoder.encoder.old_layer = runner.jiant_model.encoder.encoder.layer[:]

    task_name = datastructures.take_one(runner.jiant_model.submodels_dict.keys())
    val_results_list = []
    for i in display.trange(13):
        runner.jiant_model.encoder.encoder.layer = runner.jiant_model.encoder.encoder.old_layer[:i]
        val_results_dict = runner.run_val(
            task_name_list=runner.jiant_task_container.task_run_config.val_task_list,
        )
        val_results_list.append(val_results_dict[task_name]["metrics"].major)
    io.write_json(val_results_list, path=args.output_path)


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
