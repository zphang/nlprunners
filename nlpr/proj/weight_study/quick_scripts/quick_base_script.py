import pyutils.io as io
import nlpr.proj.weight_study.split_dict as split_dict
import torch
import os
import glob
import numpy as np
import pyutils.display as display


def flatten_dict(d):
    return np.concatenate([v.reshape(-1) for v in d.values()])


def load_split(base_path, key_list):
    return {
        key: flatten_dict(torch.load(os.path.join(base_path, key), map_location="cpu"))
        for key in key_list
    }


def main():
    path_ls = io.read_json("/home/zp489/scratch/working/v1/2003/30_cosine_similarity/metadata/path_ls.json")
    key_list = [
        'encoder.embeddings.',
        'encoder.encoder.layer.0.',
        'encoder.encoder.layer.1.',
        'encoder.encoder.layer.2.',
        'encoder.encoder.layer.3.',
        'encoder.encoder.layer.4.',
        'encoder.encoder.layer.5.',
        'encoder.encoder.layer.6.',
        'encoder.encoder.layer.7.',
        'encoder.encoder.layer.8.',
        'encoder.encoder.layer.9.',
        'encoder.encoder.layer.10.',
        'encoder.encoder.layer.11.',
    ]
    base_weights = load_split("/home/zp489/scratch/working/v1/2004/01_cosine/data/base", key_list)

    result = {key: np.zeros(810) for key in key_list}
    for i, path in enumerate(display.tqdm(path_ls[:5])):
        other_weights = load_split(path, key_list)
        for key in key_list:
            result[key][i] = (base_weights[key] * other_weights[key]).sum()
    torch.save(result, "/home/zp489/scratch/working/v1/2004/01_cosine/data/base_partials/all.p")


main()
