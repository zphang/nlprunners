import numpy as np
import torch
import os
import tqdm
import scipy.sparse
from sklearn.random_projection import GaussianRandomProjection

import pyutils.io as io
import zconf

import nlpr.proj.weight_study.split_dict as split_dict


def do_projection(split_dict_key, spec, path_ls, proj_size):
    projection = GaussianRandomProjection(proj_size, random_state=spec["seed"])
    dummy_mat = scipy.sparse.csr_matrix((1, spec["size"]))
    projection.fit(dummy_mat)
    result_dict = {}
    for path in tqdm.tqdm(path_ls):
        loaded = split_dict.load_split_dict(
            path,
            prefix_ls=[split_dict_key]
        )
        stacked = np.concatenate([v.reshape(-1) for v in loaded.values()])
        result = projection.transform(stacked[spec["slice"][0]: spec["slice"][1]].reshape(1, -1))
        result_dict[path] = result
    return result_dict


def load_spec_dict(path):
    loaded = torch.load(path)
    loaded_spec_dict = {}
    for k, v in loaded["seeds"].items():
        for i, seed in enumerate(v):
            buffer_slice = loaded["buffer_slice_dict"][k][i]
            loaded_spec_dict[(k, i)] = {
                "seed": seed,
                "slice": buffer_slice,
                "proj_size": loaded["proj_size"],
                "size": buffer_slice[1] - buffer_slice[0]
            }
    return loaded_spec_dict


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    run_i = zconf.attr(type=int, required=True)
    spec_dict_path = zconf.attr(type=str, required=True)
    path_ls_path = zconf.attr(type=str, required=True)
    output_base_path = zconf.attr(type=str, required=True)

    # === Optional === #
    buffer_size = zconf.attr(type=int, default=100000)
    proj_size = zconf.attr(type=int, default=500)
    spec_only = zconf.attr(action="store_true")


def project_and_save(run_i, spec_dict_path, path_ls_path, output_base_path):
    os.makedirs(output_base_path, exist_ok=True)
    spec_dict = load_spec_dict(spec_dict_path)
    spec_dict_key_list = list(spec_dict.keys())
    split_dict_key, split_dict_key_i = spec_dict_key = spec_dict_key_list[run_i]
    path_ls = io.read_json(path_ls_path)
    result = do_projection(
        split_dict_key=split_dict_key,
        spec=spec_dict[spec_dict_key],
        path_ls=path_ls,
        proj_size=spec_dict[spec_dict_key]["proj_size"],
    )
    os.makedirs(os.path.join(output_base_path, split_dict_key), exist_ok=True)
    torch.save(result, os.path.join(output_base_path, split_dict_key, f"{split_dict_key_i}.p"))


def main(args: RunConfiguration):
    project_and_save(
        run_i=args.run_i,
        spec_dict_path=args.spec_dict_path,
        path_ls_path=args.path_ls_path,
        output_base_path=args.output_base_path,
    )


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
