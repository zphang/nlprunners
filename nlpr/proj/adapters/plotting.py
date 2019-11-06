import colorsys
import random

import pandas as pd
import matplotlib.pyplot as plt


def generate_n_colors(n, seed=1):
    random.seed(seed)
    rgb_ls = []
    for i in range(n):
        h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
        r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
        rgb_ls.append((r / 256, g / 256, b / 256))
    return rgb_ls


def flatten_axes(axes):
    if len(axes.shape) == 1:
        return axes
    else:
        return [ax for sub_axes in axes for ax in sub_axes]


def plot_single(lines_ax, bars_ax, loaded, layer_key, color_srs):
    plot_df = pd.DataFrame(
        data=[d["weights"][layer_key] for d in loaded],
        index=[d['tgs']["global_step"] for d in loaded],
    )
    plot_df.plot(ax=lines_ax, color=color_srs[plot_df.columns].values, legend=False)
    lines_ax.set_xlim(plot_df.index[0], plot_df.index[-1] + 1)
    plot_df.iloc[-1].plot(kind="barh", ax=bars_ax, color=color_srs[plot_df.columns].values)
    bars_ax.grid()


def plot_all(loaded, color_srs, compute_keys=False):
    if compute_keys:
        seen = set()
        layer_keys = []
        for k, v in loaded[0]["weights"].items():
            hashed = str(v)
            if hashed not in seen:
                layer_keys.append(k)
            seen.add(hashed)
    else:
        layer_keys = loaded[0]["weights"].keys()
    num_sets = len(layer_keys)
    if num_sets == 4:
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        layer_keys = list(loaded[0]["weights"].keys())
        axes = [
            [axes[0][0], axes[0][1]],
            [axes[0][2], axes[0][3]],
            [axes[1][0], axes[1][1]],
            [axes[1][2], axes[1][3]],
        ]
        for i, (ax1, ax2) in enumerate(axes):
            plot_single(ax1, ax2, loaded, layer_keys[i], color_srs)
            ax1.set_title(layer_keys[i])
        plt.tight_layout()
    elif num_sets == 1:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        ax1, ax2 = axes
        plot_single(ax1, ax2, loaded, list(layer_keys)[0], color_srs)
        plt.tight_layout()
    else:
        raise KeyError(num_sets)
    return fig


def get_color_srs(components):
    return pd.Series(dict(zip(components, generate_n_colors(len(components)))))
