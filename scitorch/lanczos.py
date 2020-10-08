from typing import Callable

import numpy as np
from tqdm import tqdm
import torch


def reorthogonalize(v: torch.Tensor, B: torch.Tensor):
    assert v.size(0) == B.size(0)
    overlap = v @ B
    _v = v - (overlap.unsqueeze(0) * B).sum(dim=-1)
    return _v


def lanczos(mat_vec_mul_closure: Callable[[torch.Tensor], torch.Tensor],
            num_params: int,
            num_iter: int,
            orthogonalize: bool = False):
    
    v_j = torch.randn(num_params)
    v_j_1 = torch.zeros_like(v_j)
    v_j /= v_j.norm(p=2)
    
    T = torch.zeros((num_iter, num_iter))
    V = torch.zeros((num_params, num_iter))
    for i in range(num_iter):
        if i == 0:
            w_j = mat_vec_mul_closure(v_j)
            alpha_j = w_j.T @ v_j
            w_j = w_j - alpha_j * v_j
            T[i, i] = alpha_j
            V[:, i] = v_j.squeeze()
        else:
            v_j_1 = v_j
            w_j_1 = w_j
            
            beta_j = w_j_1.norm(p=2)
            v_j = w_j_1 / beta_j
            
            if orthogonalize:
                v_j = reorthogonalize(v_j, V)
            
            w_j = mat_vec_mul_closure(v_j)
            alpha_j = w_j.T @ v_j
            
            w_j = w_j - alpha_j * v_j - beta_j * v_j_1
            T[i, i] = alpha_j
            T[i-1, i] = beta_j
            T[i, i-1] = beta_j
            V[:, i] = v_j.squeeze()
            
    return V, T


def stochastic_lanczos_quadrature(
    mat_vec_mul_closure: Callable[[torch.Tensor], torch.Tensor],
    num_params: int,
    num_lanczos_iter: int,
    num_stochastic_iter: int,
    orthogonalize: bool = False):

    nodes_ensemble = []
    weights_ensemble = []
    for _ in tqdm(range(num_stochastic_iter), leave=False):

        _, T = lanczos(mat_vec_mul_closure,
                       num_params,
                       num_lanczos_iter,
                       orthogonalize)
        L, U = torch.eig(T, eigenvectors=True)
        L = L[:, 0]
        _nodes = L
        _weights = U[0, :]**2
        nodes_ensemble.append(_nodes.cpu().data.numpy())
        weights_ensemble.append(_weights.cpu().data.numpy())
    
    return np.asarray(nodes_ensemble), np.asarray(weights_ensemble)


def spectral_density_estimation(
    x: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
    sigma: float):

    K = nodes.shape[0]
    assert K == weights.shape[0], "Nodes and weights of the density estimator are not equal"
    _x = x[:, np.newaxis]
    _n = nodes.flatten()[np.newaxis, :]
    _w = weights.flatten()[np.newaxis, :]
    
    exponent = - 0.5 * (_x - _n)**2 / sigma**2
    spectrum = (_w * np.exp(exponent) / np.sqrt(2*np.pi * sigma**2)).sum(axis=-1) / K
    return spectrum
