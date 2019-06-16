# Differentiable sorting
A Python implementation of a fully-differentiable sorting function for power-of-2 length vectors. Uses Numpy (or PyTorch), but trivial to use in other backends.


    from differentiable_sorting import bitonic_matrices, diff_bisort

    # sort 8 element vectors
    sort_matrices = bitonic_matrices(8)
    print(diff_bisort(sort_matrices, [5.0, -1.0, 9.5, 13.2, 16.2, 20.5, 42.0, 18.0]))

    >>> [-1.007  4.996  9.439 13.212 15.948 18.21  20.602 42.   ]

## Bitonic sorting

[Bitonic sorts](https://en.wikipedia.org/wiki/Bitonic_sorter) allow creation of sorting networks with a sequence of fixed conditional swapping operations executed on an `n` element vector in parallel where, `n=2^k`.

<img src="BitonicSort1.svg.png">

*[Image: from Wikipedia, by user Bitonic, CC0](https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort1.svg)*

The sorting network for `n=2^k` elements has `k(k-1)/2` "layers" where parallel compare-and-swap operations are used to rearrange a `n` element vector into sorted order.

### Differentiable compare-and-swap

If we define the `softmax(a,b)` function (not the traditional "softmax" used for classification!) as the continuous approximation to the `max(a,b)` function, `softmax(a,b) = log(exp(a) + exp(b))`. We can then write `softmin(a,b)` as `softmin(a,b) = -log(exp(-a) + exp(-b))`

Note that we now have a differentiable compare-and-swap operation: `softrank(a,b) = (softmin(a,b), softmax(a,b))`

## Differentiable sorting

For each layer in the sorting network, we can split all of the pairwise comparison-and-swaps into left-hand and right-hand sides. We can any write function that selects the relevant elements of the vector as a multiply with a binary matrix.

For each layer, we can derive two binary matrices `left` and `right` which select the elements to be compared for the left and right hands respectively. This will result in the comparison between two `n/2` length vectors. We can also derive two matrices `l_inv` and `r_inv` which put the results of the compare-and-swap operation back into the right positions in the original vector.

The entire sorting network can then be written in terms of matrix multiplies and the `softrank(a, b)` operation.

```python
        def softmax(a, b):
            return np.log(np.exp(a) + np.exp(b))

        def softmin(a, b):
            return -softmax(-a, -b)

        def softrank(a, b):
            return softmin(a, b), softmax(a, b)

        def diff_bisort(matrices, x):   
            for l, r, l_inv, r_inv in matrices:
                a, b = softrank(l @ x, r @ x)
                x = l_inv @ a + r_inv @ b
            return x
```

The rest of the code is simply computing the `l, r, l_inv, r_inv` matrices.
