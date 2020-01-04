import zconf

import nlpr.testing_tools.example_processing as example_processing
import nlpr.shared.path_utils as path_utils


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    model_config_base_path = zconf.attr(type=str, required=True)
    task_config_base_path = zconf.attr(type=str, default=None)
    output_base_path = zconf.attr(type=str, required=True)
    reference_base_path = zconf.attr(type=str, default=None)


def get_default_reference_base():
    return path_utils.get_nlpr_path(["test_data", "example_processing"])


def get_default_task_config_base_path():
    return path_utils.get_nlpr_path(["test_data", "example_processing", "inputs", "task_configs"])


def run_checks(args: RunConfiguration):
    if args.reference_base_path is None:
        reference_base_path = get_default_reference_base()
    else:
        reference_base_path = args.reference_base_path

    if args.task_config_base_path is None:
        task_config_base_path = get_default_task_config_base_path()
    else:
        task_config_base_path = args.task_config_base_path

    example_processing.write_out(
        model_config_base_path=args.model_config_base_path,
        task_config_base_path=task_config_base_path,
        output_base_path=args.output_base_path,
    )
    example_processing.run_checks(
        base_path_1=args.output_base_path,
        base_path_2=reference_base_path,
    )


def main():
    args = RunConfiguration.run_cli()
    run_checks(args)


if __name__ == "__main__":
    main()
