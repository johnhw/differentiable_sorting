import autograd.numpy as np
from autograd import grad, jacobian
from differentiable_sorting import bitonic_matrices, diff_sort, diff_argsort
from differentiable_sorting import (
    bitonic_indices,
    diff_sort_indexed,
    diff_argsort_indexed,
    vector_sort,
)
from differentiable_sorting import softmax, smoothmax, softmax_smooth

maxes = [softmax, smoothmax, softmax_smooth]

np.random.seed(2019)


def test_vector_sort():
    jac_vector_sort = jacobian(vector_sort, argnum=1)

    for n in [2, 4, 8, 16, 32]:
        matrices = bitonic_matrices(n)
        for d in [1, 2, 4, 8]:
            X = np.random.uniform(-100, 100, (d, n))
            weight = np.random.uniform(0, 1, d)
            jac = jac_vector_sort(matrices, X, lambda x: (x.T @ weight).T)


def test_jacobian_sort():
    for n in [2, 4, 8, 16, 32]:
        matrices = bitonic_matrices(n)
        jac_sort = jacobian(diff_sort, argnum=1)
        jac_rank = jacobian(diff_argsort, argnum=1)
        vec = np.random.uniform(-200, 200, n)
        for max_fn in maxes:
            jac_vec = jac_sort(matrices, vec, max_fn)
            jac_r = jac_rank(matrices, vec, softmax=max_fn, sigma=0.1)

