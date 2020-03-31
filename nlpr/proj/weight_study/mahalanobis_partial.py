import numpy as np
import tqdm
import os

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
    end_a = zconf.attr(type=int, required=True)
    segment_size = zconf.attr(type=int, default=None)
    output_base_path = zconf.attr(type=str, required=True)


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


def get_arr(path_ls, prefix, start_i, segment_size):
    arr = None
    i_range = range(start_i, start_i + segment_size)
    for i, idx in enumerate(tqdm.tqdm(i_range, desc="Loading curr")):
        loaded = flatten_dict(split_dict.load_split_dict(path_ls[idx], prefix_ls=[prefix]))
        if arr is None:
            arr = np.empty([segment_size, len(loaded)], dtype=np.float32)
        arr[i] = loaded.astype(np.float32)
    return arr


def compute_mahalanobis_partial_many_vs_one(tensor_a, tensor_b, variances):
    normalized_sum_of_squares_tensor = tensor_a.clone()
    normalized_sum_of_squares_tensor -= tensor_b
    normalized_sum_of_squares_tensor *= normalized_sum_of_squares_tensor  # no in-place pow
    normalized_sum_of_squares_tensor /= variances
    normalized_sum_of_squares = normalized_sum_of_squares_tensor.sum(1).cpu().numpy()
    del normalized_sum_of_squares_tensor
    return normalized_sum_of_squares


def row_compute_mahalanobis_partial(path_ls, prefix, start_a, segment_size, variances):
    n = len(path_ls)
    arr_a = get_arr(
        path_ls=path_ls,
        prefix=prefix,
        start_i=start_a,
        segment_size=segment_size,
    )
    tensor_a = torch.from_numpy(arr_a).cuda()
    variances = torch.FloatTensor(variances).cuda()
    normalized_sum_of_squares_arr = np.zeros([segment_size, n])
    for idx_b in tqdm.trange(start_a, len(path_ls)):
        arr_b = flatten_dict(split_dict.load_split_dict(path_ls[idx_b], prefix_ls=[prefix]))
        tensor_b = torch.FloatTensor(arr_b).cuda().view(1, -1)

        normalized_sum_of_squares_arr[:, idx_b] = compute_mahalanobis_partial_many_vs_one(
            tensor_a=tensor_a,
            tensor_b=tensor_b,
            variances=variances,
        )

    return {
        "normalized_sum_of_squares": normalized_sum_of_squares_arr,
        "num": arr_a.shape[1],
        "start_a": start_a,
    }


def main(args: RunConfiguration):
    path_ls = io.read_json(args.path_ls_path)
    variances = load_variances(
        squared_differences_path=args.squared_differences_path,
        prefix=args.prefix,
    )
    os.makedirs(args.output_base_path, exist_ok=True)
    for start_a in range(args.start_a, args.end_a, args.segment_size):
        sub_results = row_compute_mahalanobis_partial(
            path_ls=path_ls,
            prefix=args.prefix,
            start_a=start_a,
            segment_size=args.segment_size,
            variances=variances,
        )
        torch.save(sub_results, os.path.join(args.output_base_path, f"part___{start_a:05d}.p"))


if __name__ == "__main__":
    main(args=RunConfiguration.default_run_cli())
