# Differentiable parallel approximate sorting networks
A Python implementation of a fully-differentiable *approximate* sorting function for power-of-2 length vectors. Uses Numpy (or autograd, JAX or PyTorch), but trivial to use in other backends. Works on GPU. 
```python

    from differentiable_sorting import bitonic_matrices, diff_bisort, diff_argsort

    # sort 8 element vectors
    sort_matrices = bitonic_matrices(8)
    x = [5.0, -1.0, 9.5, 13.2, 16.2, 20.5, 42.0, 18.0]
    print(diff_bisort(sort_matrices, x)) 

    >>> [-1.007  4.996  9.439 13.212 15.948 18.21  20.602 42.   ]

    from differentiable_sorting import diff_bisort_smooth
    # we can relax towards averaging
    x = [5.0, -1.0, 9.5, 13.2, 16.2, 20.5, 42.0, 18.0]
    print(diff_bisort_smooth(sort_matrices, x, smooth=0.0)) # as above

    >>> [-1.007  4.996  9.439 13.212 15.948 18.21  20.602 42.   ]

    print(diff_bisort_smooth(sort_matrices, x, smooth=0.05)) # smoothed

    >>> [ 1.242  5.333  9.607 12.446 16.845 18.995 20.932 37.999]
        
    print(diff_bisort_smooth(sort_matrices, x, smooth=1.0)) # relax completely to mean

    >>> [15.425 15.425 15.425 15.425 15.425 15.425 15.425 15.425]

    print(np.mean(x))

    >>> 15.425

    ###### Ranking
    # We can rank as well
    x = [1, 2, 3, 4, 8, 7, 6, 4]
    print(diff_argsort(sort_matrices, x))

    >>> [0. 1. 2. 3. 7. 6. 5. 3.]

    # smoothed ranking function
    print(diff_argsort(sort_matrices, x, sigma=5))

    >>> [2.8  3.   3.2  3.41 4.24 4.04 3.83 3.41]

```

## Bitonic sorting

[Bitonic sorts](https://en.wikipedia.org/wiki/Bitonic_sorter) allow creation of sorting networks with a sequence of fixed conditional swapping operations executed on an `n` element vector in parallel where, `n=2^k`.

<img src="BitonicSort1.svg.png">

*[Image: from Wikipedia, by user Bitonic, CC0](https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort1.svg)*

The sorting network for `n=2^k` elements has `k(k-1)/2` "layers" where parallel compare-and-swap operations are used to rearrange a `n` element vector into sorted order.



### Differentiable compare-and-swap

If we define the `softmax(a,b)` function (not the traditional "softmax" used for classification!) as the continuous approximation to the `max(a,b)` function, `softmax(a,b) = log(exp(a) + exp(b))`. We can then write `softmin(a,b)` as `softmin(a,b) = -log(exp(-a) + exp(-b))`

Note that we now have a differentiable compare-and-swap operation: `softcswap(a,b) = (softmin(a,b), softmax(a,b))`

This idea was inspired by [this tweet](https://twitter.com/francoisfleuret/status/1139580698694733825) by @francoisfleuret:
> François Fleuret @francoisfleuret Jun 14
>
>Discussion with Ronan Collober reminded me that (max, +) is a semi-ring, and made me realize that the same is true for (softmax, +) where
>
>softmax(a, b) = log(exp(a) + exp(b))
>
>All this coolness at the heart of his paper 
>
>https://arxiv.org/abs/1902.06022 

## Differentiable sorting

For each layer in the sorting network, we can split all of the pairwise comparison-and-swaps into left-hand and right-hand sides. We can any write function that selects the relevant elements of the vector as a multiply with a binary matrix.

For each layer, we can derive two binary matrices `left` and `right` which select the elements to be compared for the left and right hands respectively. This will result in the comparison between two `n/2` length vectors. We can also derive two matrices `l_inv` and `r_inv` which put the results of the compare-and-swap operation back into the right positions in the original vector.

The entire sorting network can then be written in terms of matrix multiplies and the `softcswap(a, b)` operation.

```python
        def softmax(a, b):
            return np.log(np.exp(a) + np.exp(b))

        def softmin(a, b):
            return -softmax(-a, -b)

        def softcswap(a, b):
            return softmin(a, b), softmax(a, b)

        def diff_bisort(matrices, x):   
            for l, r, l_inv, r_inv in matrices:
                a, b = softcswap(l @ x, r @ x)
                x = l_inv @ a + r_inv @ b
            return x
```

The rest of the code is simply computing the `l, r, l_inv, r_inv` matrices, which are fixed for a given `n`.
