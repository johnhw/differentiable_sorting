import torch
from ..differentiable_sorting import diff_sort
from ..differentiable_sorting import bitonic_matrices as np_bitonic_matrices

### Softmax (log-sum-exp)
def softmax(a, b, alpha=1, normalize=0):
    """The softmaximum of softmax(a,b) = log(e^a + a^b).
    normalize should be zero if a or b could be negative and can be 1.0 (more accurate)
    if a and b are strictly positive.
    """
    return torch.log(torch.exp(a * alpha) + torch.exp(b * alpha) - normalize) / alpha


### Smooth max
def smoothmax(a, b, alpha=1):
    return (a * torch.exp(a * alpha) + b * torch.exp(b * alpha)) / (
        torch.exp(a * alpha) + torch.exp(b * alpha)
    )


### relaxed softmax
def softmax_smooth(a, b, smooth=0):
    """The smoothed softmaximum of softmax(a,b) = log(e^a + a^b).
    With smooth=0.0, is softmax; with smooth=1.0, averages a and b"""
    t = smooth / 2.0
    return torch.log(
        torch.exp((1 - t) * a + b * t) + torch.exp((1 - t) * b + t * a)
    ) - torch.log(1 + smooth)


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

