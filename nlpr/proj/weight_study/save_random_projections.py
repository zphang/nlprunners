import math
import numpy as np
import os
import tqdm

import zconf
import torch
import scipy.sparse
from sklearn.random_projection import GaussianRandomProjection

MAX_INT = np.iinfo(np.int32).max


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    sizes_dict_path = zconf.attr(type=str, required=True)
    seed = zconf.attr(type=int, required=True)
    output_path = zconf.attr(type=str, required=True)

    # === Optional === #
    buffer_size = zconf.attr(type=int, default=100000)
    proj_size = zconf.attr(type=int, default=500)
    spec_only = zconf.attr(action="store_true")


def generate_gaussian_matrices(sizes_dict, buffer_size, proj_size, seed, output_path, spec_only=False):
    os.makedirs(output_path, exist_ok=True)
    rng = np.random.RandomState(seed)
    seed_dict = {}
    path_dict = {}
    buffer_slice_dict = {}
    for k, size in tqdm.tqdm(sizes_dict.items()):
        os.makedirs(os.path.join(output_path, k), exist_ok=True)
        n_buffer = math.ceil(size / buffer_size)
        seed_list = rng.randint(MAX_INT, size=n_buffer)
        path_list = []
        buffer_slice_ls = []
        for i in tqdm.trange(n_buffer):
            path = os.path.join(output_path, k, f"{i}")
            path_list.append(path + ".npy")
            dummy_buffer_slice = ((i * buffer_size), min((i + 1) * buffer_size, size))
            dummy_buffer_size = dummy_buffer_slice[1] - dummy_buffer_slice[0]
            buffer_slice_ls.append(dummy_buffer_slice)
            if not spec_only:
                projection = GaussianRandomProjection(proj_size, random_state=seed_list[i])
                dummy_mat = scipy.sparse.csr_matrix((1, dummy_buffer_size))
                projection.fit(dummy_mat)
                np.save(path, np.array(projection.components_).astype(np.float32))
        seed_dict[k] = seed_list
        path_dict[k] = path_list
        buffer_slice_dict[k] = buffer_slice_ls
    metadata = {
        "seeds": seed_dict,
        "sizes": sizes_dict,
        "path_dict": path_dict,
        "buffer_slice_dict": buffer_slice_dict,
        "initial_seed": seed,
        "proj_size": proj_size,
    }
    torch.save(metadata, os.path.join(output_path, "metadata.p"))


def main(args: RunConfiguration):
    generate_gaussian_matrices(
        sizes_dict=torch.load(args.sizes_dict_path),
        buffer_size=args.buffer_size,
        proj_size=args.proj_size,
        seed=args.seed,
        output_path=args.output_path,
        spec_only=args.spec_only,
    )


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
