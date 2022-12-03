#
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import pytest
import torch
from sklearn.cluster import KMeans as sklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from torch_kmeans import KMeans

DEFAULT_PARAMS = {
    "init_method": "rnd",
    "num_init": 8,
    "max_iter": 100,
    "p_norm": 2,
    "tol": 1e-4,
    "normalize": None,
    "n_clusters": 8,
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
        ("normalize", False),
        ("normalize", True),
    ],
)
def test_params(key, val):
    kwargs = deepcopy(DEFAULT_PARAMS)
    kwargs[key] = val
    KMeans(**kwargs)


@pytest.mark.parametrize(
    "key,val",
    [
        ("init_method", "abc"),
        ("num_init", 0),
        ("num_init", -1),
        ("max_iter", 0),
        ("max_iter", -1),
        ("p_norm", 0),
        ("p_norm", -1),
        ("tol", -1),
        ("tol", 2),
        ("normalize", "abc"),
        ("n_clusters", -1),
        ("n_clusters", 0),
        ("n_clusters", 1),
    ],
)
def test_params_raise(key, val):
    kwargs = deepcopy(DEFAULT_PARAMS)
    kwargs[key] = val
    with pytest.raises(ValueError):
        KMeans(**kwargs)


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
    model = KMeans(max_iter=10)
    res = model(x, k)
    assert res is not None


def test_input_data2():
    x, y, k, w = get_data(bs=3, n=20, d=2, k=2)
    model = KMeans(verbose=True)
    res = model(x, k)
    assert res is not None
    model = KMeans(verbose=True, tol=1e-1)
    res = model(x, k)
    assert res is not None
    model = KMeans(max_iter=10)
    res = model(x, k.unsqueeze(-1))
    assert res is not None
    model = KMeans(max_iter=10, normalize="minmax")
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
    model = KMeans(max_iter=10)
    with pytest.raises(ValueError):
        res = model(x, k)  # noqa


def test_input_data_raise2():
    x, y, k, w = get_data(bs=3, n=20, d=2, k=2)
    model = KMeans(max_iter=10)
    with pytest.raises(TypeError):
        res = model(x.numpy(), k)  # noqa
    with pytest.raises(TypeError):
        res = model(x, k.numpy())  # noqa
    with pytest.raises(TypeError):
        res = model(x, k, centers=x.numpy())  # noqa
    with pytest.raises(ValueError):
        res = model(x[0], k)  # noqa
    c = x[0]
    with pytest.raises(ValueError):
        res = model(x, k, centers=c)  # noqa
    c = x[:, 0]
    with pytest.raises(ValueError):
        res = model(x, k, centers=c)  # noqa
    c = x[:, :, 0]
    with pytest.raises(ValueError):
        res = model(x, k, centers=c)  # noqa
    with pytest.raises(ValueError):
        res = model(x, k, centers=x[:, None, :, :].expand(3, 5, 20, 2))  # noqa
    model = KMeans(max_iter=10, n_clusters=None)
    with pytest.raises(ValueError):
        res = model(x, k=None)  # noqa


@pytest.mark.parametrize(
    "key,val",
    [
        ("init_method", "rnd"),
        ("init_method", "k-means++"),
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
    x, y, k, w = get_data(bs=3, n=20, d=2, k=2, add_noise=False, seed=123)
    model = KMeans(**kwargs)
    res = model(x, k)
    for y_, lbl in zip(y, res.labels):
        assert (y_.view(-1) == lbl.view(-1)).sum() <= 1 or (
            y_.view(-1) != lbl.view(-1)
        ).sum() <= 1


def test_api():
    K = 4
    x, y, k, w = get_data(bs=3, n=20, d=2, k=K)
    torch.manual_seed(123)
    model1 = KMeans(n_clusters=K)
    assert model1.num_clusters is None
    m = model1.fit(x)
    assert m.is_fitted
    assert m.num_clusters is not None
    y1 = m.predict(x)
    torch.manual_seed(123)
    model2 = KMeans(n_clusters=K)
    y2 = model2.fit_predict(x)
    torch.manual_seed(123)
    model3 = KMeans()
    y3 = model3.fit_predict(x, k=k)
    assert (y1.view(-1) == y2.view(-1)).all()
    assert (y1.view(-1) == y3.view(-1)).all()


def test_result_sklearn():
    ITERS = 100
    SEED = 123
    torch.manual_seed(SEED)
    # for K=2 we can check labels and centers for the
    # two different cases of assigning labels (0, 1) or (1, 0)
    K = 2
    x, y, k, w = get_data(bs=3, n=20, d=2, k=K)
    model1 = KMeans(
        init_method="rnd", n_clusters=K, max_iter=ITERS, num_init=8, seed=SEED
    )
    res = model1(x)
    labels1 = res.labels.numpy()
    centers1 = res.centers.numpy()
    inertia1 = res.inertia.numpy()

    labels2, centers2, inertia2 = [], [], []
    for x_ in x.numpy():
        model2 = sklearnKMeans(
            init="random", n_clusters=K, max_iter=ITERS, n_init=8, random_state=SEED
        )
        model2 = model2.fit(x_)
        labels2.append(model2.labels_)
        centers2.append(model2.cluster_centers_)
        inertia2.append(model2.inertia_)

    for y_, l1, l2 in zip(y, labels1, labels2):
        assert (l1 == l2).sum() <= 1 or (l1 != l2).sum() <= 1

    for c1, c2 in zip(centers1, centers2):
        assert (np.isclose(c1[0], c2[0]).all() or np.isclose(c1[0], c2[1]).all()) and (
            np.isclose(c1[1], c2[1]).all() or np.isclose(c1[1], c2[0]).all()
        )

    for i1, i2 in zip(inertia1, inertia2):
        assert np.isclose(i1, i2)

    # for K>2 it is easier to compare permutation invariant measures like NMI and AMI
    K = 4
    x, y, k, w = get_data(bs=3, n=20, d=2, k=K)
    model1 = KMeans(
        init_method="rnd", n_clusters=K, max_iter=ITERS, num_init=8, seed=SEED
    )
    res = model1(x)
    labels1 = res.labels.numpy()
    inertia1 = res.inertia.numpy()

    labels2, inertia2 = [], []
    for x_ in x.numpy():
        model2 = sklearnKMeans(
            init="random", n_clusters=K, max_iter=ITERS, n_init=8, random_state=SEED
        )
        model2 = model2.fit(x_)
        labels2.append(model2.labels_)
        inertia2.append(model2.inertia_)

    for y_, l1, l2 in zip(y, labels1, labels2):
        nmi1 = normalized_mutual_info_score(y_, l1)
        ami1 = adjusted_mutual_info_score(y_, l1)
        nmi2 = normalized_mutual_info_score(y_, l2)
        ami2 = adjusted_mutual_info_score(y_, l2)
        assert np.isclose(nmi1, nmi2) and np.isclose(ami1, ami2)

    for i1, i2 in zip(inertia1, inertia2):
        assert np.isclose(i1, i2)


def test_result_sklearn_centers():
    ITERS = 100
    SEED = 123
    torch.manual_seed(SEED)
    N = 20
    BS = 3
    # for K=2 we can check labels and centers for the
    # two different cases of assigning labels (0, 1) or (1, 0)
    K = 2
    x, y, k, w = get_data(bs=BS, n=N, d=2, k=K)
    # random centers selected from x
    rnd_idx = torch.multinomial(
        torch.empty((BS, N), device=x.device, dtype=x.dtype).fill_(1 / N),
        num_samples=K,
        replacement=False,
    )
    centers = x.gather(
        index=rnd_idx.view(BS, -1)[:, :, None].expand(BS, -1, 2), dim=1
    ).view(BS, K, 2)

    model1 = KMeans(
        init_method="rnd", n_clusters=K, max_iter=ITERS, num_init=1, seed=SEED
    )
    res = model1(x, centers=centers.unsqueeze(1))
    labels1 = res.labels.numpy()
    centers1 = res.centers.numpy()
    inertia1 = res.inertia.numpy()

    labels2, centers2, inertia2 = [], [], []
    for x_, c in zip(x.numpy(), centers.numpy()):
        model2 = sklearnKMeans(
            init=c, n_clusters=K, max_iter=ITERS, n_init=1, random_state=SEED
        )
        model2 = model2.fit(x_)
        labels2.append(model2.labels_)
        centers2.append(model2.cluster_centers_)
        inertia2.append(model2.inertia_)

    for y_, l1, l2 in zip(y, labels1, labels2):
        assert (l1 == l2).sum() <= 1 or (l1 != l2).sum() <= 1

    for c1, c2 in zip(centers1, centers2):
        assert (np.isclose(c1[0], c2[0]).all() or np.isclose(c1[0], c2[1]).all()) and (
            np.isclose(c1[1], c2[1]).all() or np.isclose(c1[1], c2[0]).all()
        )

    for i1, i2 in zip(inertia1, inertia2):
        assert np.isclose(i1, i2)

    # for K>2 it is easier to compare permutation invariant measures like NMI and AMI
    K = 4
    x, y, k, w = get_data(bs=3, n=20, d=2, k=K)
    # random centers selected from x
    rnd_idx = torch.multinomial(
        torch.empty((BS, N), device=x.device, dtype=x.dtype).fill_(1 / N),
        num_samples=K,
        replacement=False,
    )
    centers = x.gather(
        index=rnd_idx.view(BS, -1)[:, :, None].expand(BS, -1, 2), dim=1
    ).view(BS, K, 2)

    model1 = KMeans(
        init_method="rnd", n_clusters=K, max_iter=ITERS, num_init=1, seed=SEED
    )
    res = model1(x, centers=centers.unsqueeze(1))
    labels1 = res.labels.numpy()
    inertia1 = res.inertia.numpy()

    labels2, inertia2 = [], []
    for x_, c in zip(x.numpy(), centers.numpy()):
        model2 = sklearnKMeans(
            init=c, n_clusters=K, max_iter=ITERS, n_init=1, random_state=SEED
        )
        model2 = model2.fit(x_)
        labels2.append(model2.labels_)
        inertia2.append(model2.inertia_)

    for y_, l1, l2 in zip(y, labels1, labels2):
        nmi1 = normalized_mutual_info_score(y_, l1)
        ami1 = adjusted_mutual_info_score(y_, l1)
        nmi2 = normalized_mutual_info_score(y_, l2)
        ami2 = adjusted_mutual_info_score(y_, l2)
        assert np.isclose(nmi1, nmi2) and np.isclose(ami1, ami2)

    for i1, i2 in zip(inertia1, inertia2):
        assert np.isclose(i1, i2)


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

    model1 = KMeans(n_clusters=K, max_iter=ITERS, seed=SEED)
    res1 = model1(x)
    model2 = KMeans(n_clusters=K, max_iter=ITERS, seed=SEED)
    res2 = model2(x.to(gpu_device))

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


# TODO: test against sklearn including normalization
