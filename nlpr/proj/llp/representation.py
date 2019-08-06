import torch
import torch.nn.functional as F
import nlpr.proj.llp.propagate as llp_propagate


def compute_global_agg_loss(embedding, label_ids, big_m_tensor, all_labels_tensor, const_tau):
    p_i_given_v = llp_propagate.compute_partial_p_i_given_v_gpu(
        m_tensor_chunk=embedding,
        big_m_tensor=big_m_tensor,
        const_tau=const_tau,
        # Maybe add exclusion?
    )
    selector = F.one_hot(all_labels_tensor).t()[label_ids].float()
    p_a_given_v = (p_i_given_v * selector).sum(dim=1)

    per_example_global_agg_loss = -torch.log(p_a_given_v)
    return per_example_global_agg_loss


def compute_global_agg_loss_v2(embedding, label_ids, big_m_tensor, all_labels_tensor,
                               batch_indices, const_tau):
    temp_big_m_tensor = big_m_tensor.clone()
    temp_big_m_tensor[batch_indices] = embedding
    p_i_given_v = llp_propagate.compute_partial_p_i_given_v_gpu(
        m_tensor_chunk=embedding,
        big_m_tensor=temp_big_m_tensor,
        const_tau=const_tau,
        # Maybe add exclusion?
    )
    selector = F.one_hot(all_labels_tensor).t()[label_ids].float()
    p_a_given_v = (p_i_given_v * selector).sum(dim=1)

    per_example_global_agg_loss = -torch.log(p_a_given_v)
    return per_example_global_agg_loss
