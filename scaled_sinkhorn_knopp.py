import torch
import random

import numpy as np


@torch.no_grad()
def scaled_sinkhorn_knopp(x, eta, beta, n_iters=50, cuda=True):
    """
    Given raw cluster prediction probabilities (x), return cluster assignments aligned with beta
    :param x: Raw cluster prediction probabilities. Shape: (N,K)
    :param eta: Exponent applied to raw predictions for scaling
    :param beta: Prescribed probability distribution to which clusters will be assigned. Shape: (K,1)
    :param n_iters: How many iterations to perform
    :param cuda: If beta should be moved to GPU
    :return: Cluster assignment (probabilities) aligned with beta. Shape: (N,K)
    """
    beta = beta.squeeze()
    if cuda:
        beta = beta.cuda()
    # Two scaling options
    Q = torch.pow(x, eta)
    # Q = torch.exp(x / 0.05)
    bsz = Q.shape[0]
    ratio = bsz / beta.sum()
    beta *= ratio

    Q /= torch.sum(Q)
    for it in range(n_iters):
        col_sum = torch.sum(Q, dim=0, keepdim=True)
        col_sum /= beta
        Q /= col_sum
        Q /= bsz

        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= bsz

    Q *= bsz

    return Q


def delete_idx(values, idx):
    return torch.cat([
        values[:idx], values[idx+1:]
    ])


def make_single_item(class_idx, k, mu):
    this_item = torch.rand(k)
    this_item[class_idx] = 0

    off_class_exp_sum = torch.exp(delete_idx(this_item, class_idx)).sum()
    target_class_value = torch.log((mu * off_class_exp_sum) / (1 - mu))
    this_item[class_idx] = target_class_value

    return this_item


def make_informed_probs(sz, k, mu, beta):
    items = []

    dist_scalar = sz / beta.sum()
    cluster_members = (beta * dist_scalar).long()
    remainder = sz - cluster_members.sum()
    assert remainder >= 0

    for _ in range(sz):
        class_idx = random.choices([i for i in range(k)], weights=beta, k=1)[0]
        if mu < 0:
            this_item = torch.rand(k)
        else:
            this_item = make_single_item(class_idx, k, mu)

        items.append(this_item.view(1, -1))

    items = torch.cat(items, dim=0)
    return items


def make_linear_sinkhorn_beta(n_classes, max_scalar, min_scalar=1):
    n_steps = n_classes - 1
    scalars = [(idx / n_steps) * min_scalar + ((n_steps - idx) / n_steps) * max_scalar for idx in range(n_classes)]
    return torch.tensor(scalars)


def make_cluster_vec(x, k):
    _, x_ = torch.topk(x, k=1, dim=-1)
    cluster_ids, cluster_counts = np.unique(x_.numpy(), return_counts=True)
    cluster_vec = torch.zeros(1, k)
    for cluster_id, cluster_count in zip(cluster_ids, cluster_counts):
        cluster_vec[0, cluster_id] = cluster_count

    return cluster_vec
