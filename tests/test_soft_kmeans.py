#
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs
from torch_kmeans import LpDistance, SoftKMeans

DEFAULT_PARAMS = {
    "init_method": "rnd",
    "num_init": 1,
    "max_iter": 100,
    "p_norm": 1,
    "tol": 1e-4,
    "normalize": "unit",
    "n_clusters": 8,
}

# raise warnings as error but ignore user warnings
pytestmark = pytest.mark.filterwarnings("error", "ignore::UserWarning")


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


# wilder et al. soft kmeans implementation
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


@pytest.mark.parametrize(
    "key,val", [("temp", 0.0), ("temp", -1.0), ("distance", LpDistance)]
)
def test_params_raise(key, val):
    kwargs = deepcopy(DEFAULT_PARAMS)
    kwargs[key] = val
    with pytest.raises(ValueError):
        SoftKMeans(**kwargs)


# bs, n, d, k, different_k, k_lims
@pytest.mark.parametrize(
    "bs,n,d,k,different_k,k_lims",
    [
        (1, 20, 1, 4, False, None),
        (2, 20, 1, 4, False, None),
        (10, 20, 1, 4, False, None),
        (4, 50, 1, 4, False, None),
        (4, 20, 2, 4, False, None),
        (4, 20, 2, 8, False, None),
        (4, 20, 2, 12, False, None),
        (4, 20, 2, 16, False, None),
        (4, 20, 10, 4, False, None),
        (4, 20, 2, 2, False, None),
        (4, 20, 2, 2, True, (2, 5)),
        (4, 20, 2, 2, True, (2, 16)),
    ],
)
def test_input_data(bs, n, d, k, different_k, k_lims):
    x, y, k, w = get_data(bs, n, d, k, different_k, k_lims)
    model = SoftKMeans(max_iter=10)
    res = model(x, k)
    assert res is not None


def test_input_data2():
    x, y, k, w = get_data(bs=2, n=20, d=2, k=2)
    model = SoftKMeans(verbose=True)
    res = model(x, k)
    assert res is not None
    model = SoftKMeans(verbose=True, tol=1e-1)
    res = model(x, k)
    assert res is not None


@pytest.mark.parametrize(
    "bs,n,d,k,different_k,k_lims",
    [
        (4, 1, 1, 2, False, None),
        (4, 2, 1, 2, False, None),
        (4, 20, 2, 1, False, None),
    ],
)
def test_input_data_raise(bs, n, d, k, different_k, k_lims):
    x, y, k, w = get_data(bs, n, d, k, different_k, k_lims)
    model = SoftKMeans(max_iter=10)
    with pytest.raises(ValueError):
        res = model(x, k)  # noqa


@pytest.mark.parametrize(
    "key,val",
    [
        ("init_method", "rnd"),
        ("num_init", 4),
        ("p_norm", 1),
        ("normalize", "mean"),
        ("normalize", "unit"),
    ],
)
def test_result(key, val):
    kwargs = deepcopy(DEFAULT_PARAMS)
    kwargs["max_iter"] = 300
    kwargs["tol"] = 1e-6
    kwargs[key] = val
    torch.manual_seed(123)
    x, y, k, w = get_data(bs=2, n=20, d=2, k=2, add_noise=False, seed=123)
    model = SoftKMeans(**kwargs)
    res = model(x, k)
    for y_, lbl in zip(y, res.labels):
        assert (y_.view(-1) == lbl.view(-1)).sum() <= 1 or (
            y_.view(-1) != lbl.view(-1)
        ).sum() <= 1


def test_result_wilder():
    """
    Compare the result with the
    original implementation of Wilder et al.
    """
    BS = 3
    N = 20
    TOL = 1e-6
    TEMP = 5.0
    ITERS = 300
    SEED = 123
    torch.manual_seed(SEED)
    # for K=2 we can check labels and centers for the
    # two different cases of assigning labels (0, 1) or (1, 0)
    K = 4
    x, y, k, w = get_data(bs=BS, n=N, d=2, k=K, add_noise=False)
    rnd_idx = torch.multinomial(
        torch.empty((BS, N), device=x.device, dtype=x.dtype).fill_(1 / N),
        num_samples=K,
        replacement=False,
    )
    centers = x.gather(
        index=rnd_idx.view(BS, -1)[:, :, None].expand(BS, -1, 2), dim=1
    ).view(BS, K, 2)

    torch.manual_seed(SEED + 1)
    model1 = SoftKMeans(tol=TOL, temp=TEMP, max_iter=ITERS)
    x1 = x.clone()
    res = model1(x1, k.clone(), centers=centers.clone())
    _, centers1, inertia1, _, _, _, soft_assign1 = res

    torch.manual_seed(SEED + 1)
    x2 = x.clone()
    centers2, inertia2, soft_assign2 = [], [], []
    for x_, k_, c_ in zip(x2, k.clone(), centers.clone()):
        mu, r, dist, mu_init = run_soft_k_means(
            x=x_, k=k_, num_iter=ITERS, temp=TEMP, centers=c_
        )
        soft_assign2.append(r)
        centers2.append(mu)
        inertia2.append(dist)

    RTOL = 1e-2
    ATOL = 1e-4
    for sa1, sa2 in zip(soft_assign1, soft_assign2):
        assert torch.allclose(sa1, sa2, rtol=RTOL, atol=ATOL)

    for c1, c2 in zip(centers1, centers2):
        assert (
            torch.allclose(c1[0], c2[0], rtol=RTOL, atol=ATOL)
            or torch.allclose(c1[0], c2[1], rtol=RTOL, atol=ATOL)
        ) and (
            torch.allclose(c1[1], c2[1], rtol=RTOL, atol=ATOL)
            or torch.allclose(c1[1], c2[0], rtol=RTOL, atol=ATOL)
        )

    for i1, i2 in zip(inertia1, inertia2):
        assert torch.allclose(i1, i2, rtol=RTOL, atol=ATOL)


def test_api():
    K = 4
    x, y, k, w = get_data(bs=3, n=20, d=2, k=K)
    torch.manual_seed(123)
    model1 = SoftKMeans(n_clusters=K)
    m = model1.fit(x)
    y1 = m.predict(x)
    torch.manual_seed(123)
    model2 = SoftKMeans(n_clusters=K)
    y2 = model2.fit_predict(x)
    torch.manual_seed(123)
    model3 = SoftKMeans()
    y3 = model3.fit_predict(x, k=k)
    assert (y1.view(-1) == y2.view(-1)).all()
    assert (y1.view(-1) == y3.view(-1)).all()


def test_CUDA():
    if not torch.cuda.is_available():
        return True
    gpu_device = torch.device("cuda")
    ITERS = 100
    SEED = 123
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    K = 2
    x, y, k, w = get_data(bs=3, n=20, d=2, k=K)

    model1 = SoftKMeans(n_clusters=K, max_iter=ITERS, seed=SEED)
    res1 = model1(x)

    model2 = SoftKMeans(n_clusters=K, max_iter=ITERS, seed=SEED)
    res2 = model2(x.to(gpu_device))

    for itm in res2:
        if itm is not None and isinstance(itm, torch.Tensor):
            assert itm.is_cuda

    # for y_, l1, l2 in zip(y, res1.labels, res2.labels.cpu()):
    #     assert (l1 == l2).sum() <= 1 or (l1 != l2).sum() <= 1

    RTOL = 1e-3
    ATOL = 1e-5
    for c1, c2 in zip(res1.centers, res2.centers.cpu()):
        assert (
            torch.allclose(c1[0], c2[0], rtol=RTOL, atol=ATOL)
            or torch.allclose(c1[0], c2[1], rtol=RTOL, atol=ATOL)
        ) and (
            torch.allclose(c1[1], c2[1], rtol=RTOL, atol=ATOL)
            or torch.allclose(c1[1], c2[0], rtol=RTOL, atol=ATOL)
        )

    for i1, i2 in zip(res1.inertia, res2.inertia.cpu()):
        assert torch.allclose(i1, i2, rtol=RTOL, atol=ATOL)


def test_gradient_propagation():
    torch.manual_seed(123)
    x, y, k, w = get_data(bs=2, n=20, d=2, k=2)
    model = SoftKMeans(**DEFAULT_PARAMS)

    x1 = x.clone()
    x1.requires_grad = True
    res = model(x1, k)
    loss = res.soft_assignment.sum()
    loss.backward()
    assert x1.grad is not None
    assert not torch.isnan(x1.grad).any()
