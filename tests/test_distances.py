#
import pytest
import torch
from torch_kmeans.utils.distances import (
    BaseDistance,
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
)


def test_base_distance():
    # wrong params
    with pytest.raises(ValueError):
        BaseDistance(normalize_embeddings=1)
    BaseDistance(p=1)
    BaseDistance(p=1.0)
    with pytest.raises(ValueError):
        BaseDistance(p="abc")
    with pytest.raises(ValueError):
        BaseDistance(p=0)
    BaseDistance(power=2)
    BaseDistance(power=2.0)
    with pytest.raises(ValueError):
        BaseDistance(power="abc")
    with pytest.raises(ValueError):
        BaseDistance(power=0)
    with pytest.raises(ValueError):
        BaseDistance(is_inverted=0)
    x = torch.ones(3, 5, 5)
    y = torch.ones(3, 5, 5) + 1
    dist = BaseDistance(p=2)
    # norm
    assert torch.allclose(dist.get_norm(x), torch.sqrt(torch.tensor([5])))
    assert torch.allclose(dist.get_norm(y), torch.sqrt(torch.tensor([20])))
    # forward etc.
    with pytest.raises(NotImplementedError):
        dist(x, y)
    with pytest.raises(NotImplementedError):
        dist.compute_mat(x, y)
    with pytest.raises(NotImplementedError):
        dist.pairwise_distance(x, y)


def test_lp_distance():
    dist = LpDistance(normalize_embeddings=False, p=2)
    # mat
    emb = torch.arange(4).float()
    cmp = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )[None, :, :].expand(4, 4, 4)
    with pytest.raises(AssertionError):
        dist.compute_mat(emb, emb)
    emb = emb[None, :].expand(4, -1)
    assert (dist.compute_mat(emb, emb) == cmp).all()
    emb = emb.unsqueeze(-1)
    assert (dist.compute_mat(emb, emb) == cmp).all()
    emb = emb.unsqueeze(-1)
    with pytest.raises(AssertionError):
        dist.compute_mat(emb, emb)
    emb = emb.squeeze(-1).to(dtype=torch.float16)
    with pytest.raises(TypeError):
        dist.compute_mat(emb, emb)
    # pairwise
    emb = torch.arange(4).float()
    assert dist.pairwise_distance(emb, emb) == 2 * 1e-6  # default eps
    emb2 = torch.arange(4, 0, -1).float()
    assert torch.isclose(
        dist.pairwise_distance(emb, emb2),
        torch.sqrt((torch.abs(emb - emb2) ** 2).sum()),
    )
    emb = emb[None, :].expand(4, -1)
    emb2 = emb2[None, :].expand(4, -1)
    assert torch.allclose(
        dist.pairwise_distance(emb, emb2),
        torch.sqrt((torch.abs(emb - emb2) ** 2).sum(-1)),
    )
    emb = torch.ones(4, 4, 2)
    emb2 = torch.zeros(4, 4, 2)
    assert torch.allclose(
        dist.pairwise_distance(emb, emb2), torch.sqrt(torch.tensor([2]))
    )


def test_dotproduct_similarity():
    dist = DotProductSimilarity()
    assert dist.is_inverted
    # mat
    emb = torch.ones(4, 5, 2)
    emb2 = torch.zeros(4, 5, 2)
    emb3 = -torch.ones(4, 5, 2)
    assert torch.allclose(dist.compute_mat(emb, emb), torch.tensor([2.0]))
    assert (dist.compute_mat(emb, emb2) == dist.compute_mat(emb2, emb)).all()
    assert torch.allclose(dist.compute_mat(emb, emb2), torch.zeros(1))
    assert torch.allclose(dist.compute_mat(emb, emb3), -torch.tensor([2.0]))
    emb4 = torch.ones(4, 8, 2)
    mat = dist.compute_mat(emb, emb4)
    assert mat.size() == torch.Size((4, emb.size(1), emb4.size(1)))
    # pairwise
    pwd = dist.pairwise_distance(emb, emb)
    assert torch.allclose(pwd, torch.tensor([2.0]))
    assert pwd.size() == torch.Size((emb.size(0), emb.size(1)))
    emb = emb[:, 0]
    pwd = dist.pairwise_distance(emb, emb)
    assert torch.allclose(pwd, torch.tensor([2.0]))
    assert pwd.size() == torch.Size((emb.size(0),))


def test_cosine_similarity():
    dist = CosineSimilarity()
    assert dist.is_inverted
    assert dist.normalize_embeddings
