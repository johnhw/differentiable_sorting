import torch
import numpy as np
from ..differentiable_sorting import diff_sort
from ..differentiable_sorting import bitonic_matrices as np_bitonic_matrices

### Softmax (log-sum-exp)
def softmax(a, b, alpha=1.0, normalize=0.0):
    """The softmaximum of softmax(a,b) = log(e^a + a^b).
    normalize should be zero if a or b could be negative and can be 1.0 (more accurate)
    if a and b are strictly positive.
    """
    return torch.log(torch.exp(a * alpha) + torch.exp(b * alpha) - normalize) / alpha


### Smooth max
def smoothmax(a, b, alpha=1.0):
    return (a * torch.exp(a * alpha) + b * torch.exp(b * alpha)) / (
        torch.exp(a * alpha) + torch.exp(b * alpha)
    )


### relaxed softmax
def softmax_smooth(a, b, smooth=0.0):
    """The smoothed softmaximum of softmax(a,b) = log(e^a + a^b).
    With smooth=0.0, is softmax; with smooth=1.0, averages a and b"""
    t = smooth / 2.0
    return torch.log(
        torch.exp((1.0 - t) * a + b * t) + torch.exp((1.0 - t) * b + t * a)
    ) - np.log(1.0 + smooth)


### differentiable ranking
def order_matrix(original, sortd, sigma=0.1):
    """Apply a simple RBF kernel to the difference between original and sortd,
    with the kernel width set by sigma. Normalise each row to sum to 1.0."""
    diff = (original.reshape(-1, 1) - sortd) ** 2
    rbf = torch.exp(-(diff) / (2 * sigma ** 2))
    return (rbf.t() / torch.sum(rbf, dim=1)).t()


def diff_argsort(matrices, x, sigma=0.1, softmax=softmax, transpose=False):
    """Return the smoothed, differentiable ranking of each element of x. Sigma
    specifies the smoothing of the ranking. """
    sortd = diff_sort(matrices, x, softmax)
    order = order_matrix(x, sortd, sigma=sigma)
    if transpose:
        order = order.t()
    return order @ (torch.arange(len(x), dtype=x.dtype))


def bitonic_matrices(n):
    matrices = np_bitonic_matrices(n)
    return [
        [torch.from_numpy(matrix).float() for matrix in matrix_set]
        for matrix_set in matrices
    ]


def vector_sort(matrices, X, key, alpha=1):
    """
    Sort a matrix X, applying a differentiable function "key" to each vector
    while sorting. Uses softmax to weight components of the matrix.
    
    For example, selecting the nth element of each vector by 
    multiplying with a one-hot vector.
    
    Parameters:
    ------------
        matrices:   the nxn bitonic sort matrices created by bitonic_matrices
        X:          an [n,d] matrix of elements
        key:        a function taking a d-element vector and returning a scalar
        alpha=1.0:  smoothing to apply; smaller alpha=smoother, less accurate sorting,
                    larger=harder max, increased numerical instability
        
    Returns:
    ----------
        X_sorted: [n,d] matrix (approximately) sorted accoring to 
        
    """
    for l, r, map_l, map_r in matrices:

        x = key(X)
        # compute weighting on the scalar function
        a, b = l @ x, r @ x
        a_weight = torch.exp(a * alpha) / (torch.exp(a * alpha) + torch.exp(b * alpha))
        b_weight = 1 - a_weight
        # apply weighting to the full vectors
        aX = l @ X
        bX = r @ X
        w_max = (a_weight * aX.T + b_weight * bX.T).T
        w_min = (b_weight * aX.T + a_weight * bX.T).T
        # recombine into the full vector
        X = (map_l @ w_max) + (map_r @ w_min)
    return X
