import pytest
import numpy as np
from differentiable_sorting import bitonic_matrices, bitonic_woven_matrices
from differentiable_sorting import softmax, smoothmax, softmax_smooth
from differentiable_sorting import diff_sort, diff_argsort

# non-power-of-2
# tests for smoothing, woven structure, argsort, etc.
np.random.seed(2019)
maxes = [np.maximum, softmax, smoothmax, softmax_smooth]


def test_network():
    for n in [2, 4, 8, 16, 32, 64, 128]:
        matrices = bitonic_matrices(n)
        for i in range(5):
            for r in [0.01, 1, 10, 100, 1000]:
                vec = np.random.uniform(-r, r, n)
                # verify exact sort is working
                assert np.allclose(np.sort(vec), diff_sort(matrices, vec, np.maximum))


def test_argsort():
    for n in [2, 4, 8, 16, 32, 64, 128]:
        matrices = bitonic_matrices(n)
        for i in range(5):
            for dtype in [np.float32, np.float64]:
                for r in [10, 200, 250]:
                    vec = np.random.uniform(-r, r, n).astype(dtype)
                    for sigma in [1e-1, 1, 10]:
                        for max_fn in maxes:
                            argsorted = diff_argsort(
                                matrices, vec, sigma, softmax=max_fn
                            )
                            assert np.all(argsorted >= 0)


def test_sort():
    for n in [2, 4, 8, 16, 32, 64, 128]:
        matrices = bitonic_matrices(n)
        for i in range(5):
            for dtype in [np.float32, np.float64]:
                for r in [10, 200, 250]:
                    vec = np.random.uniform(-r, r, n).astype(dtype)
                    for max_fn in maxes:
                        sorted_vec = diff_sort(matrices, vec, max_fn)
                        truth = np.sort(vec)
                        # check error is (roughly) bounded in this range
                        assert np.max(np.abs(truth - sorted_vec)) < r / 2


def test_softmax():
    for dtype in [np.float32, np.float64]:
        for r in [1e-2, 1, 10, 100]:
            vec_a = np.random.uniform(-r, r, 32).astype(dtype)
            vec_b = np.random.uniform(-r, r, 32).astype(dtype)
            for max_fn in maxes:
                vec_max = softmax(vec_a, vec_b)
                vec_min = vec_a + vec_b - vec_max
                assert np.all(vec_max > vec_min)


def test_matrices():
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        matrices = bitonic_matrices(n)
        assert all([len(m) == 4 for m in matrices])
        for l, r, l_i, r_i in matrices:
            # check shapes OK
            assert l.shape == (n // 2, n)
            assert r.shape == (n // 2, n)
            assert l_i.shape == (n, n // 2)
            assert r_i.shape == (n, n // 2)
            # check n elements OK
            for m in [l, r, l_i, r_i]:
                assert np.sum(m) == n // 2


def test_woven_matrices():
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        matrices = bitonic_woven_matrices(n)
        assert all([len(m) == 2 for m in matrices])
        for l, r in matrices:
            # check shapes OK
            assert l.shape == (n, n)
            assert r.shape == (n, n)
            # check n elements OK
            for m in [l, r]:
                assert np.sum(m) == n
                # test valid permutation matrix
                assert np.allclose(np.sum(m, axis=0), np.ones(n))
                assert np.allclose(np.sum(m, axis=1), np.ones(n))
