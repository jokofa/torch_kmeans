#
import os

import pytest
import torch
from torch_kmeans.utils.utils import first_nonzero, group_by_label_mean

MSG = (
    "Environment variable 'PYTORCH_JIT' must be set to '0' for testing! "
    "i.e. use 'PYTORCH_JIT=0 python -m pytest [...]'"
)


@pytest.mark.parametrize(
    "dims",
    [
        (1, 1, 20, 1),
        (1, 1, 20, 2),
        (1, 1, 100, 2),
        (4, 1, 20, 1),
        (4, 1, 20, 7),
        (4, 2, 20, 2),
        (7, 3, 4, 2),
    ],
)
def test_group_by_label_mean(dims):
    bs, m, n, d = dims
    assert n % 4 == 0
    # same value, same label
    x = torch.ones((bs, n, d))
    y = torch.zeros((bs, m, n)).long()
    k = torch.ones(bs)
    k_max = torch.arange(k.max())[None, :].expand(bs, -1).long()
    res = torch.ones(bs, m, 1, d)
    out = group_by_label_mean(x, y, k_max)
    assert list(out.shape) == [bs, m, k.max().item(), d]
    assert torch.allclose(out, res)
    # same value, different label
    idx = n // 2
    y[:, :, idx:] = 1
    k = torch.ones(bs) + 1
    k_max = torch.arange(k.max())[None, :].expand(bs, -1).long()
    out = group_by_label_mean(x, y, k_max)
    assert list(out.shape) == [bs, m, k.max().item(), d]
    assert torch.isclose(out, res)[out != 0].all()
    # different value, different label
    x[:, idx:] = 2
    res = torch.ones((bs, m, 2, d))
    res[:, :, 1] = 2
    out = group_by_label_mean(x, y, k_max)
    assert list(out.shape) == [bs, m, k.max().item(), d]
    assert torch.allclose(out, res)
    # different value, different label, different number of labels
    idx2 = int((n // 4) * 3)
    y[0, :, idx2:] = 2
    k[0] = k[0] + 1
    k_max = torch.arange(k.max())[None, :].expand(bs, -1).long()
    res = torch.ones((bs, m, 3, d))
    res[:, :, 1:] = 2
    out = group_by_label_mean(x, y, k_max)
    assert list(out.shape) == [bs, m, k.max().item(), d]
    assert torch.isclose(out, res)[out != 0].all()
    if os.environ.get("PYTORCH_JIT", "1") == "0":
        # wrong input type
        with pytest.raises(AssertionError):
            y = torch.zeros((bs, m, n)).numpy()
            group_by_label_mean(x, y, k_max)
        # wrong bs dim
        with pytest.raises(AssertionError):
            y = torch.zeros((bs + 1, m, n)).long()
            group_by_label_mean(x, y, k_max)
        # wrong n dim
        with pytest.raises(AssertionError):
            y = torch.zeros((bs, m, n + 1)).long()
            group_by_label_mean(x, y, k_max)
    else:
        raise EnvironmentError(MSG)


def test_first_nonzero():
    x1 = torch.zeros(10)
    # no non-zero element
    msk, idx = first_nonzero(x1)
    assert not msk.any()
    assert idx == 0
    # 1 non-zero element
    x1[5] = 1
    msk, idx = first_nonzero(x1)
    assert msk.all()
    assert idx == 5
    # n non-zero elements
    x1[7] = 2
    x1[9] = 3
    msk, idx = first_nonzero(x1)
    assert msk.all()
    assert idx == 5
    # all elements after idx are nonzero
    x1[5:] = 99
    msk, idx = first_nonzero(x1)
    assert msk.all()
    assert idx == 5
    # negative non-zero element
    x1 = torch.zeros(10)
    x1[5] = -1
    msk, idx = first_nonzero(x1)
    assert not msk.any()
    assert idx == 0
    # batch (1)
    x2 = torch.zeros((1, 10))
    msk, idx = first_nonzero(x2)
    assert not msk.any()
    assert idx == 0
    x2[:, 5] = 123
    msk, idx = first_nonzero(x2)
    assert msk.all()
    assert idx == 5
    # batch (4)
    x3 = torch.zeros((4, 10))
    msk, idx = first_nonzero(x3)
    assert not msk.any()
    assert (idx == 0).all()
    x3[:, 5] = 123
    msk, idx = first_nonzero(x3)
    assert msk.all()
    assert (idx == 5).all()
    # n-dim
    x4 = torch.zeros((5, 4, 3, 2))
    msk, idx = first_nonzero(x4)
    assert not msk.any()
    assert (idx == 0).all()
    x4[:, 1:] = 99
    msk, idx = first_nonzero(x4)
    assert msk[:, 1:].all()
    assert (idx == 0).all()
    x4 = torch.zeros((5, 4, 3, 2))
    x4[:, :, :, 1] = 123
    msk, idx = first_nonzero(x4)
    assert msk.all()
    assert (idx == 1).all()
    # different idx per row
    x5 = torch.zeros((4, 10))
    x5[0, 3] = 1
    x5[1, 4] = 2
    x5[2, 5] = 3
    msk, idx = first_nonzero(x5)
    assert msk[:-1].all()
    assert not msk[-1].any()
    assert (idx == torch.tensor([3, 4, 5, 0])).all()
    if os.environ.get("PYTORCH_JIT", "1") == "0":
        # wrong input type
        with pytest.raises(AssertionError):
            x = x1.numpy()
            msk, idx = first_nonzero(x)
        # wrong dim
        with pytest.raises(AssertionError):
            msk, idx = first_nonzero(x4, 2)
    else:
        raise EnvironmentError(MSG)
