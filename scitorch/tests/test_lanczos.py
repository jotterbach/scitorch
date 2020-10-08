from typing import Callable
import pytest
from typing import Callable

import numpy as np
import torch

from scitorch.lanczos import lanczos, stochastic_lanczos_quadrature, spectral_density_estimation


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


def test_spectral_density_estimator():
    """
    Uses Wigner's semi-circle distribution to test this
    """

    torch.manual_seed(42)
    D = 1000
    A = torch.randn(D, D)
    A = A + A.T
    Avp = lambda v: A @ v
    nodes, weights = stochastic_lanczos_quadrature(Avp, D, 100, 40, True)

    bins = np.linspace(-100, 100, 101)
    w = bins[1]-bins[0]
    cnt, edges = np.histogram(np.linalg.eigvalsh(A.data.numpy()), bins=bins)

    x = np.arange(-100, 100, .1)
    spec = spectral_density_estimation(x, nodes, weights, 1.)


    assert ((spec[np.digitize(edges, x)-1][:-1] - cnt/(w * cnt.sum()))**2).sum() < 1e-4