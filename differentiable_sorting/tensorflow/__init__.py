import tensorflow as tf
from ..differentiable_sorting import (
    diff_sort,
    bitonic_matrices,
    diff_sort_indexed,
    softmax,
    smoothmax,
    softmax_smooth,
    vector_sort,
)

### differentiable ranking
def order_matrix(original, sortd, sigma=0.1):
    """Apply a simple RBF kernel to the difference between original and sortd,
    with the kernel width set by sigma. Normalise each row to sum to 1.0."""
    diff = (tf.reshape(original, (-1, 1)) - tf.reshape(sortd, (1, -1))) ** 2
    rbf = tf.exp(-(diff) / (2 * sigma ** 2))
    return tf.transpose(tf.transpose(rbf) / tf.math.reduce_sum(rbf, axis=1))


def dargsort(original, sortd, sigma, transpose=False):
    order = order_matrix(original, sortd, sigma=sigma)
    if transpose:
        order = tf.transpose(order)

    return order @ tf.reshape(
        tf.cast(tf.range(original.shape[0]), original.dtype), (-1, 1)
    )


def diff_argsort(matrices, x, sigma=0.1, softmax=softmax, transpose=False):
    """Return the smoothed, differentiable ranking of each element of x. Sigma
    specifies the smoothing of the ranking. 
    If transpose is true, returns argsort; if false, returns ranking.
    """
    sortd = diff_sort(matrices, x, softmax)
    return dargsort(x, sortd, sigma, transpose)
