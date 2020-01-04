import os


def format_path_from_env(path_str):
    return path_str.replace("$", "").format(os.environ)
