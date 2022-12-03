#
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs
from torch_kmeans import ConstrainedKMeans, CosineSimilarity
from torch_kmeans.clustering.constr_kmeans import InfeasibilityError

DEFAULT_PARAMS = {
    "init_method": "rnd",
    "num_init": 8,
    "max_iter": 100,
    "p_norm": 2,
    "tol": 1e-4,
    "n_clusters": 8,
    "n_priority_trials_before_fall_back": 5,
}


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


@pytest.mark.parametrize(
    "key,val",
    [
        ("n_priority_trials_before_fall_back", 0),
        ("n_priority_trials_before_fall_back", -1),
        ("distance", CosineSimilarity),
    ],
)
def test_params_raise(key, val):
    kwargs = deepcopy(DEFAULT_PARAMS)
    kwargs[key] = val
    with pytest.raises(ValueError):
        ConstrainedKMeans(**kwargs)


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
    model = ConstrainedKMeans(max_iter=10, n_priority_trials_before_fall_back=20)
    try:
        res = model(x, k, weights=w)
    except InfeasibilityError:
        res = 0
    assert res is not None


def test_input_data2():
    x, y, k, w = get_data(bs=3, n=20, d=2, k=2)
    model = ConstrainedKMeans(verbose=True)
    res = model(x, k, weights=w)
    assert res is not None
    model = ConstrainedKMeans(verbose=True, tol=1e-1)
    res = model(x, k, weights=w)
    assert res is not None


@pytest.mark.parametrize(
    "bs,n,d,k,different_k,k_lims",
    [
        (4, 1, 1, 2, False, None),
        (4, 2, 1, 2, False, None),
        (4, 20, 2, 1, False, None),
        (4, 20, 2, 20, False, None),
    ],
)
def test_input_data_raise(bs, n, d, k, different_k, k_lims):
    x, y, k, w = get_data(bs, n, d, k, different_k, k_lims)
    model = ConstrainedKMeans(max_iter=10)
    with pytest.raises(ValueError):
        res = model(x, k, weights=w)  # noqa


def test_input_data_raise2():
    x, y, k, w = get_data(bs=3, n=20, d=2, k=2)
    model = ConstrainedKMeans()
    with pytest.raises(TypeError):
        res = model(x, k, weights=w.numpy())  # noqa
    w_ = w[0]
    with pytest.raises(ValueError):
        res = model(x, k, weights=w_)  # noqa
    w_ = w[:, 0]
    with pytest.raises(ValueError):
        res = model(x, k, weights=w_)  # noqa
    w_ = (w * 0) - 1
    with pytest.raises(ValueError):
        res = model(x, k, weights=w_)  # noqa
    w_ = (w * 0) + 2
    with pytest.raises(ValueError):
        res = model(x, k, weights=w_)  # noqa
    w_ = w.unsqueeze(-1)
    with pytest.raises(ValueError):
        res = model(x, k, weights=w_)  # noqa
    w_ = w[0].unsqueeze(-1)
    with pytest.raises(ValueError):
        res = model(x, k, weights=w_)  # noqa


@pytest.mark.parametrize(
    "key,val",
    [
        ("init_method", "rnd"),
        ("init_method", "k-means++"),
        ("init_method", "ckm++"),
        ("num_init", 4),
        ("p_norm", 1),
        ("normalize", "mean"),
        ("normalize", "unit"),
    ],
)
def test_result(key, val):
    BS = 3
    K = 2
    kwargs = deepcopy(DEFAULT_PARAMS)
    kwargs["max_iter"] = 300
    kwargs["tol"] = 1e-6
    kwargs[key] = val
    torch.manual_seed(123)
    x, y, k, w = get_data(bs=BS, n=20, d=2, k=K, add_noise=False, seed=123)
    model = ConstrainedKMeans(**kwargs)
    res = model(x, k, weights=w)
    # check labels
    for y_, lbl in zip(y, res.labels):
        assert (y_.view(-1) == lbl.view(-1)).sum() <= 1 or (
            y_.view(-1) != lbl.view(-1)
        ).sum() <= 1
    # check constraints
    for i in range(K):
        for b_ in range(BS):
            msk = res.labels[b_] == i
            w_ = w[b_]
            w_sum = w_[msk].sum()
            assert w_sum <= 1


def test_topk():
    kwargs = deepcopy(DEFAULT_PARAMS)
    kwargs["max_iter"] = 300
    kwargs["tol"] = 1e-6
    kwargs["init_method"] = "topk"
    x, y, k, w = get_data(bs=3, n=20, d=2, k=2, add_noise=False, seed=123)
    with pytest.raises(ValueError):
        model = ConstrainedKMeans(**kwargs)
        res = model(x, k, weights=w)
    kwargs["num_init"] = 1
    model = ConstrainedKMeans(**kwargs)
    res = model(x, k, weights=w)
    assert res is not None


def test_api():
    K = 4
    x, y, k, w = get_data(bs=3, n=20, d=2, k=K)
    torch.manual_seed(123)
    model1 = ConstrainedKMeans(n_clusters=K)
    m = model1.fit(x, weights=w)
    y1 = m.predict(x, weights=w)
    torch.manual_seed(123)
    model2 = ConstrainedKMeans(n_clusters=K)
    y2 = model2.fit_predict(x, weights=w)
    torch.manual_seed(123)
    model3 = ConstrainedKMeans()
    y3 = model3.fit_predict(x, k=k, weights=w)
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

    model1 = ConstrainedKMeans(n_clusters=K, max_iter=ITERS, seed=SEED)
    res1 = model1(x, weights=w)
    model2 = ConstrainedKMeans(n_clusters=K, max_iter=ITERS, seed=SEED)
    res2 = model2(x.to(gpu_device), weights=w.to(gpu_device))

    for itm in res2:
        if itm is not None and isinstance(itm, torch.Tensor):
            assert itm.is_cuda

    for y_, l1, l2 in zip(y, res1.labels, res2.labels.cpu()):
        assert (l1 == l2).sum() <= 1 or (l1 != l2).sum() <= 1

    for c1, c2 in zip(res1.centers, res2.centers.cpu()):
        assert (torch.allclose(c1[0], c2[0]) or torch.allclose(c1[0], c2[1])) and (
            torch.allclose(c1[1], c2[1]) or torch.allclose(c1[1], c2[0])
        )

    for i1, i2 in zip(res1.inertia, res2.inertia.cpu()):
        assert torch.allclose(i1, i2)
