import shlex


def parse_cli_to_run_conf(run_conf_cls, string):
    return run_conf_cls.run_cli_json_prepend(cl_args=shlex.split(string))
