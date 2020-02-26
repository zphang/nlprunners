import numpy as np
import torch
import pandas as pd


def to_numpy(wd):
    return {
        k: v.view(-1).numpy()
        for k, v in wd.items()
    }


def triangle_zip(ls):
    ls = list(ls)
    for i in range(len(ls)):
        for j in range(i + 1, len(ls)):
            yield ls[i], ls[j]


def select_all(x):
    return True


def stack_weights(wd1, selector=select_all):
    ls = []
    for k, v in wd1.items():
        if selector(k):
            ls.append(v)
    return np.concatenate(ls)


def startswith(prefix):
    def _f(s):
        return s.startswith(prefix)

    return _f


def without(ls, sub_ls):
    return [x for x in ls if x not in sub_ls]


def select_rows_columns(df, new_index):
    return df.loc[new_index].loc[:, new_index]


def sort_by_ls(ls, order_ls):
    ordering = {k: i for i, k in enumerate(order_ls)}
    return sorted(ls, key=ordering.get)


class InputPositionLookup:
    def __init__(self, lengths):
        self.lengths = lengths
        self.zero_prefixed_cum_lengths = np.concatenate([[0], lengths.cumsum()])
        self.cum_lengths = self.zero_prefixed_cum_lengths[1:]
        self.total = self.cum_lengths[-1]

    def lookup(self, indices):
        input_indices = np.less_equal(self.cum_lengths.reshape(1, -1), indices.reshape(-1, 1)).sum(1)
        position_indices = indices - self.zero_prefixed_cum_lengths[input_indices]
        return input_indices, position_indices

    def sample(self, n, rng=None):
        if rng is None:
            rng = np.random
        elif isinstance(rng, int):
            rng = np.random.RandomState(rng)
        return self.lookup(rng.randint(self.total, size=n))

    def reverse_lookup(self, example_indices, position_indices):
        return self.zero_prefixed_cum_lengths[example_indices] + position_indices

    @classmethod
    def from_path(cls, path):
        return cls(lengths=torch.load(path))


def group_input_position_indices(input_indices, position_indices):
    df = pd.DataFrame({"inp": input_indices, "pos": position_indices})
    input_indices = []
    position_indices = []
    for label, group in df.groupby("inp")["pos"]:
        input_indices.append(label)
        position_indices.append(group.values.tolist())
    return input_indices, position_indices


def group_by_example_indices(indices_dict):
    example_key = "example"
    assert example_key in indices_dict
    df = pd.DataFrame(indices_dict)
    result_dict = {k: [] for k in indices_dict.keys()}
    non_example_keys = [k for k in indices_dict.keys() if k != example_key]
    for label, sub_df in df.groupby(example_key):
        result_dict[example_key] = label
        for k in non_example_keys:
            result_dict[k].append(sub_df[k].values.tolist())
    return result_dict
