import numpy as np
import torch
import tqdm

import pyutils.io as io
import zconf

import nlpr.proj.weight_study.split_dict as split_dict


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    path_ls_path = zconf.attr(type=str, default=None)
    key_ls_path = zconf.attr(type=str, default=None)
    squared_differences_path = zconf.attr(type=str, default=None)
    start_i = zconf.attr(type=int, required=True)
    segment_size = zconf.attr(type=int, default=None)
    output_path = zconf.attr(type=str, required=True)


def flatten_dict(d, key_ls):
    return np.concatenate([d[k].reshape(-1) for k in key_ls])


def load_means(means_path, key_ls):
    means_loaded = torch.load(means_path)
    means = (
        flatten_dict(means_loaded["sums"], key_ls=key_ls)
        / means_loaded["total"]
    )
    del means_loaded
    return means


def load_variances(squared_differences_path, key_ls):
    squared_diff_loaded = torch.load(squared_differences_path)
    variances = (
        flatten_dict(squared_diff_loaded["squared_difference_sums"], key_ls=key_ls)
        / squared_diff_loaded["total"]
    )
    del squared_diff_loaded
    return variances


def compute_mahalanobis(arr_1, arr_2, variances):
    return np.sqrt(((arr_1 - arr_2) ** 2 / variances).sum())


def batch_compute_mahalanobis(path_ls, key_ls, start_i, segment_size, variances):

    results_dict = {}

    i_range = range(start_i, start_i + segment_size)
    curr_dict = {}
    for i in tqdm.trange(start_i, start_i + segment_size, desc="Loading curr"):
        curr_dict[i] = flatten_dict(split_dict.load_split_dict(path_ls[i]), key_ls=key_ls)

    for j in tqdm.trange(len(path_ls), desc="Loading other"):
        other = None  # Lazy loading
        for i in i_range:
            if i <= j:
                # Lazy loading
                if other is None:
                    other = flatten_dict(split_dict.load_split_dict(path_ls[j]), key_ls=key_ls)
                results_dict[i, j] = compute_mahalanobis(
                    arr_1=curr_dict[i], arr_2=other, variances=variances,
                )
            else:
                results_dict[i, j] = None
        del other
    return results_dict


def main(args: RunConfiguration):
    path_ls = io.read_json(args.path_ls_path)
    key_ls = io.read_json(args.key_ls_path)
    variances = load_variances(
        squared_differences_path=args.squared_differences_path,
        key_ls=key_ls,
    )
    sub_results = batch_compute_mahalanobis(
        path_ls=path_ls,
        key_ls=key_ls,
        start_i=args.start_i,
        segment_size=args.segment_size,
        variances=variances,
    )
    torch.save(sub_results, args.output_path)


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
