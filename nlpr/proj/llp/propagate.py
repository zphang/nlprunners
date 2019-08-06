import faiss
import numpy as np
import torch
import tqdm

from pyutils.display import maybe_tqdm


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def super_cross_product(x):
    return x @ x.T


def top_k(x, k):
    assert len(x.shape) == 2
    return np.argpartition(x, -k, axis=1)[:, -k:]


def get_start_ends(start, end, size):
    length = end - start
    num_chunks = length // size
    if length % size != 0:
        num_chunks += 1
    return [(i, min(i+size, end)) for i in range(start, end, size)]


def torch_cross_product(big_m, chunk_size, device):
    big_m_tensor = torch.tensor(big_m).to(device)
    result = np.empty([big_m.shape[0], big_m.shape[0]])
    with torch.no_grad():
        for start, end in tqdm.tqdm(get_start_ends(0, big_m.shape[0], size=chunk_size)):
            partial_result = torch.mm(big_m_tensor[start:end], big_m_tensor.t())
            result[start:end] = partial_result.cpu().numpy()
        return result


def local_label_propagation(big_m, num_labeled, dim_size, const_k, const_t, const_tau):
    num_data = big_m.shape[0]
    num_unlabeled = num_data - num_labeled

    index = faiss.IndexFlatIP(dim_size)  # build the index
    index.add(big_m)

    distances, raw_nt_indices = index.search(big_m, const_t+1)
    nt_indices = raw_nt_indices[:, 1:]  # exclude self
    assert nt_indices.shape == (num_data, const_t)

    mmt = super_cross_product(big_m)
    assert mmt.shape == (num_data, num_data)

    adj_exponentiated = np.exp((mmt - mmt.max(0)[:, np.newaxis]) / const_tau)
    # potentially mmt.max(0)[:, np.newaxis] = 1? is norm=1
    adj_exponentiated = np.clip(adj_exponentiated, a_min=1e-8, a_max=None)
    # Prevent placing probability density on self
    adj_exponentiated -= np.diag(np.diag(adj_exponentiated))
    partition_functions = adj_exponentiated.sum(1)
    assert partition_functions.shape == (num_data,)

    p_i_given_v = adj_exponentiated / partition_functions[:, np.newaxis]
    assert p_i_given_v.shape == (num_data, num_data)

    i_selector = np.repeat(np.arange(num_data)[:, np.newaxis], const_t, axis=1)
    t_closest_probs = p_i_given_v[i_selector, nt_indices]  # Check correctness
    assert t_closest_probs.shape == (num_data, const_t)
    # assert t_closest_probs[27, 14] == p_i_given_v[27, nt_indices[27, 14]]

    d_of_vi = t_closest_probs.sum(1)
    assert d_of_vi.shape == (num_data,)

    p_of_l_i_given_v = p_i_given_v / d_of_vi[np.newaxis, :]  # Check correctness
    assert p_of_l_i_given_v.shape == (num_data, num_data)

    relevant_p_of_l_i_given_v = p_of_l_i_given_v[num_labeled:, :num_labeled]
    assert relevant_p_of_l_i_given_v.shape == (num_unlabeled, num_labeled)

    capital_i = top_k(relevant_p_of_l_i_given_v, k=const_k)
    assert capital_i.shape == (num_unlabeled, const_k)
    # I is also the chosen labeled neighbors
    # assert (capital_i < num_labeled).all()

    chosen_p_of_l_i_given_v = relevant_p_of_l_i_given_v[
        np.arange(num_unlabeled)[:, np.newaxis], capital_i
    ]
    assert chosen_p_of_l_i_given_v.shape == (num_unlabeled, const_k)

    vote_weights = chosen_p_of_l_i_given_v
    assert vote_weights.shape == (num_unlabeled, const_k)

    return vote_weights, capital_i


def local_label_propagation_gpu(big_m, num_labeled, dim_size, const_k, const_t, const_tau,
                                chunk_size, device, verbose=True):
    torch.cuda.empty_cache()
    if isinstance(big_m, np.ndarray):
        big_m_tensor = torch.from_numpy(big_m).to(device)
    elif isinstance(big_m, torch.Tensor):
        big_m_tensor = big_m
        big_m = big_m.cpu().numpy()
    else:
        raise RuntimeError(type(big_m))

    num_data = big_m.shape[0]
    num_unlabeled = num_data - num_labeled

    index = faiss.IndexFlatIP(dim_size)  # build the index
    faiss_gpu = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(faiss_gpu, 0, index)
    gpu_index.add(big_m)

    distances, raw_nt_indices = gpu_index.search(big_m[:num_labeled], const_t + 1)
    nt_indices = raw_nt_indices[:, 1:]  # exclude self
    assert nt_indices.shape == (num_labeled, const_t)

    i_selector_tensor = torch.from_numpy(
        np.repeat(np.arange(chunk_size)[:, np.newaxis], const_t, axis=1)
    ).to(device)
    nt_indices_tensor = torch.from_numpy(nt_indices).to(device)

    # Result placeholders
    vote_weights = torch.zeros([num_unlabeled, const_k])
    capital_i = torch.zeros([num_unlabeled, const_k]).long()

    with torch.no_grad():
        # d_of_vi needs to be computed only up to the labeled data points
        d_of_vi = torch.zeros(num_labeled).to(device)
        start_ends_iterator = get_start_ends(0, num_labeled, size=chunk_size)

        for start, end in maybe_tqdm(start_ends_iterator, desc="Propagating[1]", verbose=verbose):
            used_chunk_size = end - start
            assert used_chunk_size <= chunk_size
            partial_p_i_given_v = compute_partial_p_i_given_v_gpu_lowmem(
                m_tensor_chunk=big_m_tensor[start:end],
                big_m_tensor=big_m_tensor,
                const_tau=const_tau,
                exclude_start=start,
            )
            assert partial_p_i_given_v.shape == (used_chunk_size, num_data)
            partial_t_closest_probs = partial_p_i_given_v[
                i_selector_tensor[:used_chunk_size],
                nt_indices_tensor[start:end],
            ]
            assert partial_t_closest_probs.shape == (used_chunk_size, const_t)

            partial_d_of_vi = partial_t_closest_probs.sum(1)
            assert partial_d_of_vi.shape == (used_chunk_size,)
            d_of_vi[start:end] = partial_d_of_vi

        # We only need to collect votes for the unlabeled data
        for start, end in maybe_tqdm(get_start_ends(num_labeled, big_m.shape[0], size=chunk_size),
                                     verbose=verbose):
            used_chunk_size = end - start
            assert used_chunk_size <= chunk_size
            partial_p_i_given_v = compute_partial_p_i_given_v_gpu_lowmem(
                m_tensor_chunk=big_m_tensor[start:end],
                big_m_tensor=big_m_tensor,
                const_tau=const_tau,
                exclude_start=start,
            )

            # Now we only care about probability of choosing a labeled datapoint
            partial_p_of_l_i_given_v = partial_p_i_given_v[:, :num_labeled] / d_of_vi.reshape(1, -1)
            assert partial_p_of_l_i_given_v.shape == (used_chunk_size, num_labeled)

            partial_vote_weights, partial_capital_i = partial_p_of_l_i_given_v.topk(
                k=const_k, dim=-1, sorted=False,
            )
            assert partial_vote_weights.shape == (used_chunk_size, const_k)
            assert partial_capital_i.shape == (used_chunk_size, const_k)
            vote_weights[start-num_labeled:end-num_labeled] = partial_vote_weights
            capital_i[start-num_labeled:end-num_labeled] = partial_capital_i

        torch.cuda.empty_cache()


        # I'm deciding to manually cap vote weights
        vote_weights.clamp_max_(1)

        return (
            vote_weights.cpu().numpy(),
            capital_i.cpu().numpy(),
        )


def compute_partial_p_i_given_v_gpu(m_tensor_chunk, big_m_tensor, const_tau, exclude_start=None):
    used_chunk_size = m_tensor_chunk.shape[0]
    num_data = big_m_tensor.shape[0]

    partial_mmt = torch.mm(m_tensor_chunk, big_m_tensor.t())
    assert partial_mmt.shape == (used_chunk_size, num_data,)

    partial_adj_exponentiated = torch.exp(
        (partial_mmt - partial_mmt.max(dim=1)[0].reshape(-1, 1)) / const_tau
    ).clamp(1e-8, None)
    if exclude_start is not None:
        partial_adj_exponentiated[:, exclude_start:exclude_start+used_chunk_size] -= \
            torch.diag(torch.diag(
                partial_adj_exponentiated[:, exclude_start:exclude_start+used_chunk_size]
            ))
    partial_partition_functions = partial_adj_exponentiated.sum(1)  # Can get quite high
    assert partial_partition_functions.shape == (used_chunk_size,)

    partial_p_i_given_v = partial_adj_exponentiated / partial_partition_functions.reshape(-1, 1)
    assert partial_p_i_given_v.shape == (used_chunk_size, num_data)
    return partial_p_i_given_v


def compute_partial_p_i_given_v_gpu_lowmem(m_tensor_chunk, big_m_tensor, const_tau,
                                           exclude_start=None):
    used_chunk_size = m_tensor_chunk.shape[0]
    num_data = big_m_tensor.shape[0]

    x = torch.mm(m_tensor_chunk, big_m_tensor.t())
    assert x.shape == (used_chunk_size, num_data,)
    # x = partial_mmt

    x -= x.max(dim=1)[0].reshape(-1, 1)
    x /= const_tau
    x.exp_()
    x.clamp_min_(1e-8)
    if exclude_start is not None:
        # Prevent placing probability density on self
        x[:, exclude_start:exclude_start+used_chunk_size] -= \
            torch.diag(torch.diag(x[:, exclude_start:exclude_start+used_chunk_size]))
    # x = partial_adj_exponentiated

    x /= x.sum(1).reshape(-1, 1)
    assert x.shape == (used_chunk_size, num_data)
    # partial_p_i_given_v

    return x


def compute_pseudolabels(true_labels, vote_weights, capital_i, num_classes):
    tallied_votes = np.stack([
        ((true_labels[capital_i] == k) * vote_weights).sum(1)
        for k in range(num_classes)
    ], axis=1)
    pseudolabels = np.argmax(tallied_votes, axis=-1)
    confidence = tallied_votes[np.arange(len(pseudolabels)), pseudolabels]

    # Also clamping confidence
    confidence = np.clip(confidence, a_min=None, a_max=1)
    return pseudolabels, confidence
