import os

import torch

import pyutils.strings as strings
import pyutils.io as io


def split_dict_by_prefix(d, prefix_ls, safety_check=True):
    if safety_check:
        for i in range(len(prefix_ls)):
            for j in range(i + 1, len(prefix_ls)):
                assert not prefix_ls[i].startswith(prefix_ls[j])
                assert not prefix_ls[j].startswith(prefix_ls[i])

    out_d = {}
    for k, v in d.items():
        for prefix in prefix_ls:
            if k.startswith(prefix):
                if prefix not in out_d:
                    out_d[prefix] = {}
                out_d[prefix][strings.remove_prefix(k, prefix)] = v
                break
        else:
            raise ValueError(f"'{k}' did not match any prefix")
    return out_d


def save_split_dict(split_dict, path):
    os.makedirs(path, exist_ok=True)
    assert "split_dict_metadata.json" not in split_dict
    for k, v in split_dict.items():
        torch.save(v, os.path.join(path, k))
    metadata = {
        "prefixes": {
            k: [sub_k for sub_k in v]
            for k, v in split_dict.items()
        }
    }
    io.write_json(metadata, os.path.join(path, "split_dict_metadata.json"))


def load_split_dict(path, prefix_ls=None):
    if path.endswith("split_dict_metadata.json"):
        load_base_path = os.path.split(path)[0]
        metadata = io.read_json(path)
    else:
        load_base_path = path
        metadata = io.read_json(os.path.join(path, "split_dict_metadata.json"))
    loaded = {}
    if prefix_ls is None:
        prefix_ls = metadata["prefixes"].keys()
    for k in prefix_ls:
        sub_loaded = torch.load(os.path.join(load_base_path, k))
        for sub_k, sub_v in sub_loaded.items():
            loaded[f"{k}{sub_k}"] = sub_v
    return loaded


def replace_with_split_dict(path, new_path, prefix_ls):
    loaded = torch.load(path, map_location="cpu")
    split_dict = split_dict_by_prefix(
        d=loaded,
        prefix_ls=prefix_ls,
        safety_check=True,
    )
    save_split_dict(
        split_dict=split_dict,
        path=new_path,
    )
    os.remove(path)
