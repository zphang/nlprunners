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
    prefix = zconf.attr(type=str, default=None)
    start_a = zconf.attr(type=int, required=True)
    end_a = zconf.attr(type=int, required=True)
    segment_size = zconf.attr(type=int, default=None)
    base_path = zconf.attr(type=str, default=None)
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
    for i, idx in enumerate(tqdm.trange(start_i, start_i + segment_size, desc="Loading curr")):
        loaded = flatten_dict(split_dict.load_split_dict(path_ls[idx], prefix_ls=[prefix]))
        if arr is None:
            arr = np.empty([segment_size, len(loaded)], dtype=np.float32)
        arr[i] = loaded.astype(np.float32)
    return arr


def row_compute_dot_product_partial(path_ls, prefix, start_a, segment_size, base=None):
    n = len(path_ls)
    actual_segment_size = min(start_a + segment_size, len(path_ls)) - start_a
    arr_a = get_arr(
        path_ls=path_ls,
        prefix=prefix,
        start_i=start_a,
        segment_size=actual_segment_size,
    )
    if base is not None:
        arr_a -= base
    tensor_a = torch.from_numpy(arr_a).cuda()
    sum_of_products_arr = np.zeros([actual_segment_size, n])
    for idx_b in tqdm.trange(start_a, len(path_ls)):
        arr_b = flatten_dict(split_dict.load_split_dict(path_ls[idx_b], prefix_ls=[prefix]))
        arr_b -= base
        tensor_b = torch.FloatTensor(arr_b).cuda().view(1, -1)
        sum_of_products_arr[:, idx_b] = (tensor_a * tensor_b).sum(1).cpu().numpy()

    return {
        "sum_of_products": sum_of_products_arr,
        "num": arr_a.shape[1],
        "start_a": start_a,
    }


def main(args: RunConfiguration):
    path_ls = io.read_json(args.path_ls_path)
    os.makedirs(args.output_base_path, exist_ok=True)
    if args.base_path:
        base = flatten_dict(split_dict.load_split_dict(args.base_path, prefix_ls=[args.prefix]))
    else:
        base = None
    for start_a in range(args.start_a, min(args.end_a, len(path_ls)), args.segment_size):
        sub_results = row_compute_dot_product_partial(
            path_ls=path_ls,
            prefix=args.prefix,
            start_a=start_a,
            segment_size=args.segment_size,
            base=base,
        )
        torch.save(sub_results, os.path.join(args.output_base_path, f"part___{start_a:05d}.p"))


if __name__ == "__main__":
    main(args=RunConfiguration.default_run_cli())
