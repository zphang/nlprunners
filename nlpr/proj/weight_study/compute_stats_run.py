import torch
import tqdm

import pyutils.io as io
import zconf

import nlpr.proj.weight_study.split_dict as split_dict


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    path_ls_path = zconf.attr(type=str, required=True)
    key_ls_path = zconf.attr(type=str, required=True)
    stat = zconf.attr(type=str, required=True)
    output_path = zconf.attr(type=str, required=True)

    # === Optional === #
    other_data = zconf.attr(type=str, default=None)


def main(args):
    path_ls = io.read_json(args.path_ls_path)
    key_ls = io.read_json(args.key_ls_path)
    if args.stat == "mean":
        compute_mean(
            path_ls=path_ls,
            key_ls=key_ls,
            output_path=args.output_path,
        )
    if args.stat == "squared_difference":
        compute_squared_difference(
            path_ls=path_ls,
            key_ls=key_ls,
            other_data=io.read_json(args.other_data),
            output_path=args.output_path,
        )
    else:
        raise KeyError(args.stat)


def compute_mean(path_ls, key_ls, output_path):
    sums = {}
    for path in tqdm.tqdm(path_ls):
        state_dict = split_dict.load_split_dict(path)
        for k in key_ls:
            flat = state_dict[k].reshape(-1).numpy()
            if k in sums:
                sums[k] += flat
            else:
                sums[k] = flat
    result = {
        "sums": sums,
        "total": len(path_ls),
    }
    torch.save(result, output_path)


def compute_squared_difference(path_ls, key_ls, other_data, output_path):
    means_data = torch.load(other_data["means_data"])
    means = {k: v/means_data["total"] for k, v in means_data["sums"].items()}
    squared_difference_sums = {}
    for path in tqdm.tqdm(path_ls):
        state_dict = split_dict.load_split_dict(path)
        for k in key_ls:
            flat = state_dict[k].reshape(-1).numpy()
            squared_difference = (flat - means[k]) ** 2
            if k in squared_difference_sums:
                squared_difference_sums[k] += squared_difference
            else:
                squared_difference_sums[k] = squared_difference
    result = {
        "squared_difference_sums": squared_difference_sums,
        "total": len(path_ls),
    }
    torch.save(result, output_path)


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())