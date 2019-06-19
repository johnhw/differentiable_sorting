import pytest

try:
    import torch

    print("Torch imported")
except:
    pytest.skip(
        "Could not import torch; skipping torch tests.", allow_module_level=True
    )


from differentiable_sorting.torch import bitonic_matrices, diff_sort
from differentiable_sorting.torch import softmax, smoothmax, softmax_smooth
from differentiable_sorting.torch import diff_argsort
from torch.autograd import Variable
import numpy as np


def jacobian(result, input):
    jac = []
    for i in range(len(result)):
        jac.append(torch.autograd.grad(result[i], input, retain_graph=True)[0])
    return jac


def test_sorting():
    # convert to torch tensors

    torch_matrices = bitonic_matrices(8)
    for max_fn in [softmax, smoothmax, softmax_smooth]:
        for i in range(5):
            test_input = np.random.randint(-200, 200, 8)
            test = Variable(torch.from_numpy(test_input).float(), requires_grad=True)

            result = diff_sort(torch_matrices, test, softmax=max_fn)
            jacobian(result, test)
            ranked_result = diff_argsort(torch_matrices, test, softmax=max_fn)
            jacobian(ranked_result, test)
            diff_argsort(torch_matrices, test, transpose=True, softmax=max_fn)
            argsorted_result = diff_argsort(torch_matrices, test, softmax=max_fn)
            jacobian(argsorted_result, test)


test_sorting()
