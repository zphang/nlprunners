import pyutils.io as io
import torch
import tqdm
import torch
import os
from nlpr.proj.weight_study.mahalanobis_partial_v2 import *
import torch
import zconf

import nlpr.proj.weight_study.split_dict as split_dict


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    prefix = zconf.attr(type=str, default=None)
    output_path = zconf.attr(type=str, required=True)


def main(args: RunConfiguration):
    path_ls = io.read_json("/home/zp489/scratch/working/v1/2002/22_random_projection/metadata/path_ls.json")

    base_split_dict_path = "/home/zp489/scratch/working/v1/2003/18_mahalanobis_viz/data/base/"
    base_arr = flatten_dict(split_dict.load_split_dict(base_split_dict_path, prefix_ls=[args.prefix]))

    variances = load_variances(
        squared_differences_path="/home/zp489/scratch/working/v1/2003/13_mahalanobis/stats/squared_difference_2.p",
        prefix=args.prefix,
    )
    variances = torch.FloatTensor(variances).cuda()
    tensor_b = torch.FloatTensor(base_arr).cuda().view(1, -1)

    normalized_sum_of_squares_arr = np.empty([1200])
    segment_size = 20
    for start_a in tqdm.tqdm(range(0, 1200, segment_size), "A iteration"):
        arr_a = get_arr(
            path_ls=path_ls,
            prefix=args.prefix,
            start_i=start_a,
            segment_size=segment_size,
        )
        normalized_sum_of_squares_arr[start_a: start_a + segment_size] = \
            compute_mahalanobis_partial_many_vs_one(
                tensor_a=torch.from_numpy(arr_a).cuda(),
                tensor_b=tensor_b,
                variances=variances,
            )
    result = {
        "normalized_sum_of_squares": normalized_sum_of_squares_arr,
        "num": tensor_b.shape[1],
    }
    torch.save(result, args.output_path)


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
