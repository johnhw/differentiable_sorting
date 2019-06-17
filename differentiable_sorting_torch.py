import torch

from differentiable_sorting import softcswap_smooth, diff_bisort, diff_bisort_smooth, diff_rank, bitonic_matrices

def softmax(a, b):
    """The softmaximum of softmax(a,b) = log(e^a + a^b)."""
    return torch.log(torch.exp(a) + torch.exp(b))

def softcswap(a, b):
    """Return a,b in 'soft-sorted' order, with the smaller value first"""    
    return -softmax(-a,-b), softmax(a, b)

def diff_bisort(matrices, x):
    """
    Approximate differentiable sort. Takes a set of bitonic sort matrices generated by bitonic_matrices(n), sort 
    a sequence x of length n. Values may be distorted slightly but will be ordered.
    """
    for l, r, map_l, map_r in matrices:
        a, b = softcswap(l @ x, r @ x)
        x = map_l @ a + map_r @ b
    return x

### apply relaxation to the softmax function
def softmax_smooth(a, b, smooth=0):
    """The smoothed softmaximum of softmax(a,b) = log(e^a + a^b).
    With smooth=0.0, is softmax; with smooth=1.0, averages a and b"""
    t = smooth / 2.0
    return torch.log(torch.exp((1 - t) * a + b * t) + torch.exp((1 - t) * b + t * a)) - torch.log(
        1 + smooth
    )


### differentiable ranking
def order_matrix(original, sortd, sigma=0.1):
    """Apply a simple RBF kernel to the difference between original and sortd,
    with the kernel width set by sigma. Normalise each row to sum to 1.0."""
    diff = (original - sortd.reshape(-1,1)) ** 2    
    rbf = torch.exp(-(diff) / (2 * sigma ** 2))
    return (rbf.t() / torch.sum(rbf, dim=1)).t()


def diff_rank(matrices, x, sigma=0.1):
    """Return the smoothed, differentiable ranking of each element of x. Sigma
    specifies the smoothing of the ranking. """
    sortd = diff_bisort(matrices, x)
    return order_matrix(x, sortd, sigma=sigma) @ (torch.arange(len(x), dtype=torch.float).to(x.get_device()))

