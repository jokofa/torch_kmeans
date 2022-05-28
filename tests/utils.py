#
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.datasets import make_blobs


def get_data(
    bs: int = 3,
    n: int = 20,
    d: int = 2,
    k: Optional[int] = 4,
    different_k: bool = False,
    k_lims: Optional[Tuple[int, int]] = (2, 5),
    add_noise: bool = True,
    fp_dtype=torch.float32,
    seed: int = 42,
):
    torch.manual_seed(seed)
    if different_k:
        a, b = k_lims
        k = torch.randint(low=a, high=b, size=(bs,)).long()
    else:
        k = torch.empty(bs).fill_(k).long()

    # generate pseudo clustering data
    x, y = [], []
    for i, k_ in enumerate(k.numpy()):
        x_, y_ = make_blobs(
            n_samples=n, centers=k_, n_features=d, random_state=seed + i
        )
        x.append(x_)
        y.append(y_)
    x = torch.from_numpy(np.stack(x, axis=0))
    y = torch.from_numpy(np.stack(y, axis=0))
    if add_noise:
        x += torch.randn(x.size())
    # sample weights
    weights = torch.abs(torch.randn(size=y.size()))
    # normalize weights per cluster given by label
    norm_weights = torch.empty(y.size())
    for i in range(bs):
        w = weights[i]
        y_ = y[i]
        unq = len(torch.unique(y_))
        nw = torch.empty(y_.size())
        for j in range(unq):
            msk = y_ == j
            w_ = w[msk]
            nw[msk] = w_ / (w_.sum() * 1.15)
        norm_weights[i] = nw
    assert (norm_weights.sum(dim=-1).long() <= k).all()

    return x.to(fp_dtype), y, k, norm_weights.to(dtype=fp_dtype)


def soft_k_means(x, k, num_iter, centers=None, cluster_temp=5):
    """Original implementation of Wilder et al."""
    # normalize x so it lies on the unit sphere
    x = torch.diag(1.0 / torch.norm(x, p=2, dim=1)) @ x
    if centers is None:
        # NOTE: changed original sklearn based kmeans++ init to random init:
        n, d = x.size()
        rnd_idx = torch.multinomial(
            torch.empty((n), device=x.device, dtype=x.dtype).fill_(1 / n),
            num_samples=k,
            replacement=False,
        )
        centers = x.gather(index=rnd_idx[:, None].expand(-1, d), dim=0)
    mu = centers
    for t in range(num_iter):
        # get distances between all data points and cluster centers
        dist = x @ mu.t()
        # cluster responsibilities via softmax
        r = torch.softmax(cluster_temp * dist, 1)
        # total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        # mean of points in each cluster weighted by responsibility
        cluster_mean = (r.t().unsqueeze(1) @ x.expand(k, *x.shape)).squeeze(1)
        # update cluster means
        new_mu = torch.diag(1 / cluster_r) @ cluster_mean
        mu = new_mu
    dist = x @ mu.t()
    r = torch.softmax(cluster_temp * dist, 1)
    return mu, r, dist


def run_soft_k_means(x, k, num_iter, temp, centers=None):
    # run until fixpoint
    mu_init, _, _ = soft_k_means(x, k, num_iter, cluster_temp=temp, centers=centers)
    # final propagation
    mu, r, dist = soft_k_means(
        x, k, 1, cluster_temp=temp, centers=mu_init.detach().clone()
    )
    return mu, r, dist, mu_init
