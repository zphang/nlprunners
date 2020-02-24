import numpy as np


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


from sklearn.random_projection import GaussianRandomProjection
