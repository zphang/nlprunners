import pyutils.io as io
import pyutils.display as display
import pyutils.strings as strings
import zconf

import nlpr.proj.weight_study.split_dict as split_dict


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    path_ls = zconf.attr(type=str, required=True)
    prefix_ls = zconf.attr(type=str, default=None)


def main(args: RunConfiguration):
    path_ls = io.read_json(args.path_ls)
    prefix_ls = io.read_json(args.prefix_ls)

    for path in display.tqdm(path_ls):
        new_path = strings.replace_suffix(path, ".p", "__split_dict")
        split_dict.replace_with_split_dict(
            path=path,
            new_path=new_path,
            prefix_ls=prefix_ls,
        )


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
