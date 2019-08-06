import numpy as np
import torch

import nlpr.proj.llp.propagate as propagate


def test_cpu_equals_gpu():
    num_data = 5000
    num_labeled = 200
    dim_size = 128
    const_k = 10
    const_t = 25
    const_tau = 1
    chunk_size = 1000

    big_m = np.random.randn(num_data, dim_size).astype('float32')
    big_m = big_m / np.linalg.norm(big_m, axis=1)[:, np.newaxis]

    vote_weights, capital_i = propagate.local_label_propagation(
        big_m=big_m,
        num_labeled=num_labeled,
        dim_size=dim_size, const_k=const_k, const_t=const_t, const_tau=const_tau,
    )

    vote_weights2, capital_i2 = propagate.local_label_propagation_gpu(
        big_m=big_m,
        num_labeled=num_labeled,
        dim_size=dim_size, const_k=const_k, const_t=const_t, const_tau=const_tau,
        chunk_size=chunk_size,
        device=torch.device("cuda:0"),
    )

    for i in range(4800):
        sorted_index = np.argsort(capital_i[i])
        sorted_index2 = np.argsort(capital_i2[i])
        assert (capital_i[i, sorted_index] == capital_i2[i, sorted_index2]).all()
        assert np.isclose(
            vote_weights[i, sorted_index],
            vote_weights2[i, sorted_index2],
        ).all()
