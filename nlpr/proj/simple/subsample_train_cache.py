import os

import zconf
import pyutils.io as io
import pyutils.sampling as sampling

import nlpr.shared.caching as shared_caching


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    cache_fol_path = zconf.attr(type=str, required=True)
    sample_num = zconf.attr(type=int)
    output_dir = zconf.attr(type=str, required=True)

    # === Optional parameters === #
    seed = zconf.attr(None)
    force_overwrite = zconf.attr(action="store_true")
    allow_subsample = zconf.attr(action="store_true")
    allow_supersample = zconf.attr(action="store_true")
    verbose = zconf.attr(action="store_true")


def main(args: RunConfiguration):
    train_cache = shared_caching.ChunkedFilesDataCache(
        cache_fol_path=args.cache_fol_path,
    )
    indices = sampling.safe_sample_indices(
        data_length=len(train_cache),
        n=args.sample_num,
        allow_subsample=args.allow_subsample,
        allow_supersample=args.allow_supersample,
        rng=args.seed,
    )
    data = train_cache.load_from_indices(
        indices=indices,
        verbose=True,
    )
    if not args.force_overwrite:
        assert not os.path.exists(args.output_dir)
    shared_caching.chunk_and_save(
        data=data,
        chunk_size=train_cache.chunk_size,
        data_args=train_cache.data_args,
        output_dir=args.output_dir,
    )
    io.write_json(
        data=[int(i) for i in indices],
        path=os.path.join(args.output_dir, "indices.json"),
    )


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
