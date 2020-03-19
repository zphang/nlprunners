import numpy as np
import tqdm

import torch

import pyutils.io as io
import zconf

import nlpr.proj.weight_study.split_dict as split_dict


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    path_ls_path = zconf.attr(type=str, default=None)
    squared_differences_path = zconf.attr(type=str, default=None)
    prefix = zconf.attr(type=str, default=None)
    start_a = zconf.attr(type=int, required=True)
    start_b = zconf.attr(type=int, required=True)
    segment_size = zconf.attr(type=int, default=None)
    output_path = zconf.attr(type=str, required=True)


def flatten_dict(d):
    return np.concatenate([v.reshape(-1) for v in d.values()])


def load_variances(squared_differences_path, prefix):
    squared_diff_loaded = torch.load(squared_differences_path)
    squared_difference_sums = {
        k: v
        for k, v in squared_diff_loaded["squared_difference_sums"].items()
        if k.startswith(prefix)
    }
    variances = flatten_dict(squared_difference_sums) / squared_diff_loaded["total"]
    del squared_diff_loaded
    return variances


def compute_mahalanobis_partial(arr_1, arr_2, variances):
    return {
        "normalized_sum_of_squares": ((arr_1 - arr_2) ** 2 / variances).sum(),
        "num": len(arr_1),
    }


def batch_compute_mahalanobis_partial(path_ls, prefix, start_a, start_b, segment_size, variances):

    results_dict = {}

    curr_dict = {}
    range_a = range(start_a, start_a + segment_size)
    for idx_a in tqdm.tqdm(range_a, desc="Loading curr"):
        loaded = split_dict.load_split_dict(path_ls[idx_a], prefix_ls=[prefix])
        curr_dict[idx_a] = flatten_dict(loaded)

    range_b = range(start_b, start_b + segment_size)
    for idx_b in tqdm.tqdm(range_b, desc="Loading curr"):
        loaded_b = None  # Lazy loading
        for idx_a in range_a:
            if idx_a <= idx_b:
                # Lazy loading
                if loaded_b is None:
                    if idx_b is curr_dict:
                        loaded_b = curr_dict[idx_b]
                    else:
                        loaded_b = flatten_dict(split_dict.load_split_dict(path_ls[idx_b], prefix_ls=[prefix]))
                results_dict[idx_a, idx_b] = compute_mahalanobis_partial(
                    arr_1=curr_dict[idx_a], arr_2=loaded_b, variances=variances,
                )
            else:
                results_dict[idx_a, idx_b] = None
        del loaded_b
    return results_dict


def main(args: RunConfiguration):
    path_ls = io.read_json(args.path_ls_path)
    variances = load_variances(
        squared_differences_path=args.squared_differences_path,
        prefix=args.prefix,
    )
    sub_results = batch_compute_mahalanobis_partial(
        path_ls=path_ls,
        prefix=args.prefix,
        start_a=args.start_a,
        start_b=args.start_b,
        segment_size=args.segment_size,
        variances=variances,
    )
    torch.save(sub_results, args.output_path)


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
