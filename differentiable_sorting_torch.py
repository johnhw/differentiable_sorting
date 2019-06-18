import torch
from differentiable_sorting import dsort


def softmax(a, b):
    """The softmaximum of softmax(a,b) = log(e^a + a^b)."""
    return torch.log(torch.exp(a) + torch.exp(b))


### apply relaxation to the softmax function
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


def diff_argsort(matrices, x, sigma=0.1, softmax=softmax):
    """Return the smoothed, differentiable ranking of each element of x. Sigma
    specifies the smoothing of the ranking. """
    sortd = dsort(matrices, x, softmax)
    return order_matrix(x, sortd, sigma=sigma) @ (
        torch.arange(len(x), dtype=torch.float).to(x.get_device())
    )

