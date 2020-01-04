import os


def format_path_from_env(path_str):
    return path_str.replace("$", "").format(os.environ)


def get_nlpr_base_path():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))


def get_nlpr_path(path_stubs):
    return os.path.abspath(os.path.join(get_nlpr_base_path(), *path_stubs))
