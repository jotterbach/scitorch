from typing import Callable
import pytest
from typing import Callable

import numpy as np
import torch

from scitorch.lanczos import lanczos


def test_lanczos():
    torch.manual_seed(42)

    D = 100
    A = torch.randn(D, D)
    A = A + A.T

    Av_prod: Callable = lambda v: A @ v

    _, T = lanczos(Av_prod, num_params=D, num_iter=D, orthogonalize=True)

    L, U = torch.eig(T, eigenvectors=True)

    torch.testing.assert_allclose(U @ torch.diag(L[:, 0]) @ U.T, T,
                                  rtol=1e-3, atol=1e-4)

    EV, _ = torch.eig(A)
    np.testing.assert_array_almost_equal(np.sort(L[:, 0].data.numpy()),
                                         np.sort(EV[:, 0].data.numpy()),
                                         decimal=4)

