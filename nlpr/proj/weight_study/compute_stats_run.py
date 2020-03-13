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


def main(args):
    path_ls = io.read_json(args.path_ls_path)
    key_ls = io.read_json(args.key_ls_path)
    if args.stat == "mean":
        compute_mean(
            path_ls=path_ls,
            key_ls=key_ls,
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


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())