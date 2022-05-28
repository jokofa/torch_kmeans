#
from copy import deepcopy

import pytest
import torch
from torch_kmeans import KNN, CosineSimilarity, DotProductSimilarity

from tests.utils import get_data

DEFAULT_PARAMS = {
    "k": 4,
    "p_norm": 2,
}


@pytest.mark.parametrize(
    "key,val",
    [
        ("normalize", False),
        ("normalize", True),
        ("normalize", None),
    ],
)
def test_params(key, val):
    kwargs = deepcopy(DEFAULT_PARAMS)
    kwargs[key] = val
    KNN(**kwargs)


@pytest.mark.parametrize(
    "key,val",
    [
        ("normalize", "abc"),
        ("k", 0),
        ("k", -1),
        ("p_norm", 0),
        ("p_norm", -1),
    ],
)
def test_params_raise(key, val):
    kwargs = deepcopy(DEFAULT_PARAMS)
    kwargs[key] = val
    with pytest.raises(ValueError):
        KNN(**kwargs)


@pytest.mark.parametrize(
    "bs,n,d,k,same_source",
    [
        (1, 20, 1, 4, False),
        (2, 20, 1, 4, False),
        (10, 20, 1, 4, False),
        (4, 50, 1, 4, False),
        (4, 20, 2, 4, False),
        (4, 20, 2, 8, False),
        (4, 20, 2, 12, False),
        (4, 20, 2, 16, False),
        (4, 20, 10, 4, False),
        (4, 20, 2, 2, False),
        (4, 20, 2, 2, True),
        (4, 20, 2, 2, True),
    ],
)
def test_input_data(bs, n, d, k, same_source):
    x, y, _, w = get_data(bs, n, d, k)
    model = KNN(**DEFAULT_PARAMS)
    res = model(x, k, same_source=same_source)
    assert res is not None


@pytest.mark.parametrize(
    "bs,n,d,k",
    [
        (4, 1, 1, 2),
        (4, 2, 1, 2),
        (4, 20, 2, 20),
    ],
)
def test_input_data_raise(bs, n, d, k):
    x, y, _, w = get_data(bs, n, d, k)
    model = KNN(k=k)
    with pytest.raises(ValueError):
        res = model(x, k)  # noqa


def test_input_data_raise2():
    K = 4
    x, y, k, w = get_data(bs=3, n=20, d=2, k=K)
    model = KNN(K)
    with pytest.raises(TypeError):
        res = model(x.numpy())  # noqa
    with pytest.raises(ValueError):
        res = model(x, k.numpy())  # noqa
    with pytest.raises(ValueError):
        res = model(x[0], k)  # noqa


@pytest.mark.parametrize(
    "key,val",
    [
        ("p_norm", 1),
        ("normalize", "mean"),
        ("normalize", "minmax"),
        ("normalize", "unit"),
        ("distance", DotProductSimilarity),
        ("distance", CosineSimilarity),
    ],
)
def test_result(key, val):
    K = 2
    kwargs = deepcopy(DEFAULT_PARAMS)
    kwargs[key] = val
    torch.manual_seed(123)
    x, y, _, w = get_data(bs=3, n=20, d=2, k=K, add_noise=False, seed=123)
    model = KNN(**kwargs)
    res = model(x, K)  # noqa


def test_api():
    K = 4
    x, y, _, w = get_data(bs=3, n=20, d=2, k=K)
    torch.manual_seed(123)
    model1 = KNN(k=K)
    res1 = model1.fit(x)
    torch.manual_seed(123)
    model2 = KNN(k=K)
    res2 = model2(x)
    assert (res1.distances.reshape(-1) == res2.distances.reshape(-1)).all()
    assert (res1.indices.reshape(-1) == res2.indices.reshape(-1)).all()


def test_CUDA():
    if not torch.cuda.is_available():
        return True
    gpu_device = torch.device("cuda")
    SEED = 123
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    K = 2
    x, y, _, w = get_data(bs=3, n=20, d=2, k=K)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model1 = KNN(k=K)
    res1 = model1(x)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model2 = KNN(k=K)
    res2 = model2(x.to(gpu_device))

    for itm in res2:
        if itm is not None and isinstance(itm, torch.Tensor):
            assert itm.is_cuda

    assert torch.allclose(res1.distances.reshape(-1), res2.distances.cpu().reshape(-1))
    assert torch.allclose(res1.indices.reshape(-1), res2.indices.cpu().reshape(-1))
