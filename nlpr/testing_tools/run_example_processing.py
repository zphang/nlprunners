import os

import zconf

import nlpr.testing_tools.example_processing as example_processing


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    model_config_base_path = zconf.attr(type=str, required=True)
    task_config_base_path = zconf.attr(type=str, required=True)
    output_base_path = zconf.attr(type=str, required=True)
    reference_base_path = zconf.attr(type=str, default=None)


def get_default_examples_path():
    return os.path.abspath(os.path.join(
        os.path.abspath(__file__), "..", "..", "..", "test_data", "example_processing",
    ))


def run_checks(args: RunConfiguration):
    if args.reference_base_path is None:
        reference_base_path = get_default_examples_path()
    else:
        reference_base_path = args.reference_base_path
    example_processing.write_out(
        model_config_base_path=args.model_config_base_path,
        task_config_base_path=args.task_config_base_path,
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
