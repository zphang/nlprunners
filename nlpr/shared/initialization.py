import json
import numpy as np
import os
import random
import torch

import nlpr.shared.log_info as log_info


def quick_init(args, verbose=True):
    log_info.print_args(args)
    init_server_logging(server_ip=args.server_ip, server_port=args.server_port, verbose=verbose)
    device, n_gpu = init_cuda_from_args(
        no_cuda=args.no_cuda,
        local_rank=args.local_rank,
        fp16=args.fp16,
        verbose=verbose,
    )
    args.seed = init_seed(given_seed=args.seed, n_gpu=args.n_gpu)
    init_output_dir(output_dir=args.output_dir, force_overwrite=args.force_overwrite)
    save_args(args=args, verbose=verbose)
    return device, n_gpu


def init_server_logging(server_ip, server_port, verbose=True):
    if server_ip and server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        if verbose: print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(server_ip, server_port), redirect_output=True)
        ptvsd.wait_for_attach()


def init_cuda_from_args(no_cuda, local_rank, fp16, verbose=True):
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    if verbose:
        print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(local_rank != -1), fp16)
        )

    return device, n_gpu


def init_seed(given_seed, n_gpu):
    used_seed = get_seed(given_seed)
    random.seed(used_seed)
    np.random.seed(used_seed)
    torch.manual_seed(used_seed)
    print("Using seed: {}".format(used_seed))

    if n_gpu > 0:
        torch.cuda.manual_seed_all(used_seed)

    # MAKE SURE THIS IS SET
    return used_seed


def init_output_dir(output_dir, force_overwrite):
    if not force_overwrite \
            and (os.path.exists(output_dir) and os.listdir(output_dir)):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)


def save_args(args, verbose=True):
    formatted_args = json.dumps(vars(args), indent=2)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        f.write(formatted_args)
    if verbose:
        print(formatted_args)


def get_seed(seed):
    if seed == -1:
        return int(np.random.randint(0, 2**32 - 1))
    else:
        return seed
