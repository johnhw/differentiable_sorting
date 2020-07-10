import pytest
import numpy as np
from differentiable_sorting import (
    bitonic_matrices,
    bitonic_woven_matrices,
    bitonic_indices,
)
from differentiable_sorting import diff_sort_indexed, diff_sort_weave
from differentiable_sorting import softmax, smoothmax, softmax_smooth
from differentiable_sorting import diff_sort, diff_argsort, diff_argsort_indexed
from differentiable_sorting import vector_sort, comparison_sort

# non-power-of-2
# tests for smoothing, woven structure, argsort, etc.
np.random.seed(2019)
maxes = [np.maximum, softmax, smoothmax, softmax_smooth]


def test_vector_sort():
    test_array = np.array([[-10, 2, 30, 4, 5, 6, 7, 80], [5, 6, 7, 8, 9, 10, 11, 1]]).T
    sorted_X = vector_sort(
        bitonic_matrices(8), test_array, lambda x: x @ [1, 0], alpha=1.0
    )
    assert abs(test_array[0, 0] - -10.0) < 1.0
    assert abs(test_array[-1, 0] - 80.0) < 1.0
    assert abs(test_array[0, 1] - 5.0) < 1.0
    assert abs(test_array[-1, 1] - 1.0) < 1.0

    # check that second column not affected by small changes in first
    # which preserve order
    test_array_2 = np.array([[1, 2, 70, 4, 5, 6, 7, 120], [5, 6, 7, 8, 9, 10, 11, 1]]).T
    assert np.allclose(test_array[:, 1], test_array_2[:, 1])

    for n in [2, 4, 8, 16, 64, 256]:
        matrices = bitonic_matrices(n)
        for d in [1, 2, 3, 8, 10]:
            for alpha in [0.01, 0.1, 1.0, 10.0]:
                X = np.random.uniform(-100, 100, (n, d))
                weight = np.random.uniform(0, 1, d)
                sorted_X = vector_sort(matrices, X, lambda x: x @ weight, alpha=alpha)
                assert sorted_X.shape == X.shape



def test_comparison_sort():
    np.random.seed(24)
    
    def abs_fn(a,b):
        return np.tanh(a**2 - b**2)

    def normal_fn(a,b):
        return np.tanh(a-b)

    # simple fixed test
    matrices = bitonic_matrices(8)
       
    test_array = np.array([-10, 2, 30, 4, 5, 6, 7, 80])
    sorted_X = comparison_sort(bitonic_matrices(8), test_array, abs_fn)
    assert np.allclose(sorted_X, [ 80.,  30., -10.,   7.,   6.,   5.,   4.,   2.])
   
    # check that absolute ordering works for various sizes
    for n  in [4,8,16,32,64]:
        matrices = bitonic_matrices(n)
        x = np.random.normal(0,200,n)
        abs_sorted_xs = comparison_sort(matrices, x, abs_fn)
        normal_sorted_xs = comparison_sort(matrices, x, normal_fn)

        assert np.all(np.diff(np.abs(abs_sorted_xs))<0)
        assert np.all(np.diff(normal_sorted_xs)<0)
        assert abs_sorted_xs.shape == x.shape 
        assert normal_sorted_xs.shape == x.shape 
        assert not np.allclose(abs_sorted_xs, normal_sorted_xs)
    
    def compare_fn(l, r):                
        l = np.mean(l.reshape(l.shape[0], -1), axis=1) 
        r = np.mean(r.reshape(r.shape[0], -1), axis=1) 
        return np.tanh(l-r)

    # check tensor operations work correctly
    x = np.random.normal(0,1,(16, 13, 19))
    x = (x.T + np.random.uniform(-2, 2, 16)).T
    matrices = bitonic_matrices(16)
    sorted_xs = comparison_sort(matrices, x, compare_fn, alpha=1, scale=200)
    assert sorted_xs.shape == x.shape

   


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
                            argsorted = diff_argsort(
                                matrices, vec, sigma, softmax=max_fn, transpose=True
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
        for m in matrices:
            # check shapes OK
            assert m.shape == (n, n)
            assert np.sum(m) == n
            # test valid permutation matrix
            assert np.allclose(np.sum(m, axis=0), np.ones(n))
            assert np.allclose(np.sum(m, axis=1), np.ones(n))


from scipy.stats import rankdata


# test that ranking works
def test_ranking():
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        woven = bitonic_woven_matrices(n)
        indices = bitonic_indices(n)
        matrices = bitonic_matrices(n)
        for i in range(20):
            # test ranking
            test = np.random.randint(-200, 200, n)
            true_rank = rankdata(test)
            dargsort = diff_argsort(matrices, test, sigma=0.001, softmax=np.maximum)
            assert np.allclose(true_rank - 1, dargsort)
            dargsort = diff_argsort_indexed(
                indices, test, sigma=0.001, softmax=np.maximum
            )
            assert np.allclose(true_rank - 1, dargsort)


# check that the various different representations
# all come to the same ground truth
def test_forms():
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        woven = bitonic_woven_matrices(n)
        indices = bitonic_indices(n)
        matrices = bitonic_matrices(n)
        for i in range(20):
            test = np.random.randint(-200, 200, n)
            truth = np.sort(test)
            assert np.all(diff_sort(matrices, truth, np.maximum) == truth)
            assert np.all(diff_sort_weave(woven, truth, np.maximum) == truth)
            assert np.all(diff_sort_indexed(indices, truth, np.maximum) == truth)

