import math
import matplotlib.pyplot as plt
import os

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


def qplot_single(datum, ax=None, title=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = None
    datum.plot(alpha=0.3, ax=ax)
    datum.rolling(50).mean().plot(ax=ax)
    ax.grid()
    ax.set_title(title)
    return fig


def qplot_multi(data_df, n_cols, figsize=None, return_plots=False):
    fig, flat_axes = quick_subplots(len(data_df.columns), n_cols, figsize=figsize)
    for i, (column, ax) in enumerate(zip(data_df, flat_axes)):
        qplot_single(data_df[column], ax=ax, title=f"[{i}] {column}")
    if return_plots:
        return fig, flat_axes


def get_latest_log(path, verbose=False):
    latest_fol_name = sorted(next(os.walk(path))[1])[-1]
    chosen_path = os.path.join(path, latest_fol_name)
    if verbose:
        print(chosen_path)
    return zlogv1.load_log(chosen_path)
