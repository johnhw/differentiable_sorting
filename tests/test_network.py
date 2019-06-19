from differentiable_sorting import pretty_bitonic_network, bitonic_network

def test_pretty_bitonic_network():
    for n in [2,4,8,16,32]:
        pretty_bitonic_network(n)

def test_bitonic_network():
    for n in [2,4,8,16,32]:
        bitonic_network(n)