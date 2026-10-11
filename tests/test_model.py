import numpy as np
import pytest
import torch

from jarz.model import forces, potential
from jarz.params import Chain


@pytest.mark.parametrize("b", [0.0, 1.0])
def test_forces_are_minus_gradient(b):
    P = Chain(N=5, b=b)
    rng = np.random.default_rng(0)
    r = torch.tensor(np.cumsum(rng.normal(0.3, 0.5, (3, P.N, 2)), axis=1), requires_grad=True)
    lam = 1.7
    potential(r, lam, P).sum().backward()
    F = forces(r.detach(), lam, P)
    assert torch.allclose(F, -r.grad, atol=1e-10)
