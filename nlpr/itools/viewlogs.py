import math
import matplotlib.pyplot as plt
import os

import pyutils.io as io
import zproto.zlogv1 as zlogv1


def flatten_axes(axes):
    if len(axes.shape) == 1:
        return axes
    else:
        return [ax for sub_axes in axes for ax in sub_axes]


def quick_subplots(n_axes, n_cols, figsize=None):
    n_rows = math.ceil(n_axes / n_cols)
    if figsize is None:
        figsize = (16, 3 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    return fig, flatten_axes(axes)


def qplot_single(datum, ax=None, title=None, smooth=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = None
    datum.plot(alpha=0.3, ax=ax)
    if smooth:
        datum.rolling(smooth).mean().plot(ax=ax)
    ax.grid()
    ax.set_title(title)
    return fig


def qplot_multi(data_df, n_cols, figsize=None, return_plots=False, smooth=None):
    fig, flat_axes = quick_subplots(len(data_df.columns), n_cols, figsize=figsize)
    for i, (column, ax) in enumerate(zip(data_df, flat_axes)):
        qplot_single(data_df[column], ax=ax, title=f"[{i}] {column}", smooth=smooth)
    if return_plots:
        return fig, flat_axes


def get_latest_log_path(base_path):
    latest_fol_name = sorted(next(os.walk(base_path))[1])[-1]
    chosen_path = os.path.join(base_path, latest_fol_name)
    return chosen_path


def get_latest_log(base_path, verbose=False):
    chosen_path = get_latest_log_path(base_path)
    if verbose:
        print(chosen_path)
    return zlogv1.load_log(chosen_path)


def listify(code):
    spec_ls = []  # prefix, name, i
    glob_format_tokens = []
    for i, token in enumerate(code.split("/")):
        if "$" in token:
            split_token = token.split("$")
            assert len(split_token) == 2
            spec_ls.append(split_token + [i])
            glob_format_tokens.append(f"{split_token[0]}*")
        else:
            glob_format_tokens.append(token)
    path_ls = io.sorted_glob("/".join(glob_format_tokens))
    result_ls = []
    for path in path_ls:
        tokens = path.split("/")
        result = {
            name: tokens[i][len(prefix):]
            for prefix, name, i in spec_ls
        }
        result["path"] = path
        result_ls.append(result)
    return result_ls
