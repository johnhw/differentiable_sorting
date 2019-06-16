# Differentiable_sorting
Implements a fully-differentiable sorting function for power-of-2 length vectors. Numpy or PyTorch, but trivial to use in other backends.

    from differentiable_sorting import bitonic_matrices, diff_bisort

    # sort 8 element vectors
    sort_matrices = bitonic_matrices(8)
    print(diff_bisort(sort_matrices, [5.0, -1.0, 9.5, 13.2, 16.2, 20.5, 42.0, 18.0]))

    >>> [-1.007  4.996  9.439 13.212 15.948 18.21  20.602 42.   ]

## Bitonic sorting

[Bitonic sorts](https://en.wikipedia.org/wiki/Bitonic_sorter) allow creation of sorting networks with a sequence of fixed conditional swapping operations executed in parallel. A sorting network implements  a map from $\mathbb{R}^n \rightarrow \mathbb{R}^n$, where $n=2^k$ (sorting networks for non-power-of-2 sizes are possible but not trickier).

<img src="BitonicSort1.svg.png">

*[Image: from Wikipedia, by user Bitonic, CC0](https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort1.svg)*

The sorting network for $2^k$ elements has $\frac{k(k-1)}{2}$ "layers" where parallel compare-and-swap operations are used to rearrange a $k$ element vector into sorted order.

### Differentiable compare-and-swap

If we define the `softmax(a,b)` function (not the traditional "softmax" used for classification!) as the continuous approximation to the `max(a,b)` function:

$$\text{softmax}(a,b) = \log(e^a + e^b) \approx \max(a,b).$$

We can then fairly obviously write `softmin(a,b)` as:

$$\text{softmin}(a,b) = -\log(e^{-a} + e^{-b}) \approx \min(a,b).$$

These functions obviously aren't equal to max and min, but are relatively close, and differentiable. Note that we now have a differentiable compare-and-swap operation:

$$\text{high} = \text{softmax}(a,b), \text{low} = \text{softmin}(a,b), \text{where } \text{low}\leq \text{high}$$

## Differentiable sorting

For each layer in the sorting network, we can split all of the pairwise comparison-and-swaps into left-hand and right-hand sides which can be done simultaneously. We can any write function that selects the relevant elements of the vector as a multiply with a binary matrix.

For each layer, we can derive two binary matrices $L \in \mathbb{R}^{k \times \frac{k}{2}}$ and $R \in \mathbb{R}^{k \times \frac{k}{2}}$ which select the elements to be compared for the left and right hands respectively. This will result in the comparison between two $\frac{k}{2}$ length vectors. We can also derive two matrices $L' \in \mathbb{R}^{\frac{k}{2} \times k}$ and $R' \in \mathbb{R}^{\frac{k}{2} \times k}$ which put the results of the compare-and-swap operation back into the right positions.

Then, each layer $i$ of the sorting process is just:
$${\bf x}_{i+1} = L'_i[\text{softmin}(L_i{\bf x_i}, R_i{\bf x_i})] + R'_i[\text{softmax}(L_i{\bf x_i}, R_i{\bf x_i})]$$
$$ = L'_i\left(-\log\left(e^{-L_i{\bf x}_i} + e^{-R_i{\bf x}_i}\right)\right) +  R'_i\left(\log\left(e^{L_i{\bf x}_i} + e^{R_i{\bf x}_i}\right)\right)$$
which is clearly differentiable (though not very numerically stable -- the usable range of elements $x$ is quite limited in single float precision).

All that remains is to compute the matrices $L_i, R_i, L'_i, R'_i$ for each of the layers of the network. 
