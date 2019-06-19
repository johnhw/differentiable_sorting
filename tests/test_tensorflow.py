import pytest

try:
    import tensorflow as tf

    print("Tensorflow imported")
except:
    pytest.skip(
        "Could not import tensorflow; skipping tensorflow tests.",
        allow_module_level=True,
    )


from tensorflow.python.ops.parallel_for.gradients import jacobian
from differentiable_sorting import bitonic_matrices, diff_sort
from differentiable_sorting import softmax, smoothmax, softmax_smooth
from differentiable_sorting.tensorflow import diff_argsort
import numpy as np


def to_tf(x, dtype):
    return tf.reshape(tf.convert_to_tensor(x, dtype=dtype), (-1, 1))


def test_sorting():
    # convert to TF tensors
    dtype = tf.float64
    tf_matrices = bitonic_matrices(8)
    for max_fn in [softmax, smoothmax, softmax_smooth]:
        for i in range(5):
            test = to_tf(np.random.randint(-200, 200, 8), dtype=dtype)
            tf_output = tf.reshape(diff_sort(tf_matrices, test), (-1,))
            tf_ranks = diff_argsort(tf_matrices, test)
            tf_argsort = diff_argsort(tf_matrices, test, transpose=True)
            tf_grads = tf.squeeze(jacobian(tf_output, test))
            # compute output and gradient
            with tf.Session() as s:
                s.run((tf_output, tf_grads, tf_ranks, tf_argsort))

