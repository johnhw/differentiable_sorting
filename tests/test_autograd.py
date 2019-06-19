import autograd.numpy as np
from autograd import grad, jacobian
from differentiable_sorting import bitonic_matrices, diff_sort, diff_argsort
from differentiable_sorting import (
    bitonic_indices,
    diff_sort_indexed,
    diff_argsort_indexed,
)
from differentiable_sorting import softmax, smoothmax, softmax_smooth

maxes = [softmax, smoothmax, softmax_smooth]

np.random.seed(2019)


def test_jacobian_sort():
    for n in [2, 4, 8, 16, 32]:
        matrices = bitonic_matrices(n)
        jac_sort = jacobian(diff_sort, argnum=1)
        jac_rank = jacobian(diff_argsort, argnum=1)
        vec = np.random.uniform(-200, 200, n)
        for max_fn in maxes:
            jac_vec = jac_sort(matrices, vec, max_fn)
            jac_r = jac_rank(matrices, vec, softmax=max_fn, sigma=0.1)

