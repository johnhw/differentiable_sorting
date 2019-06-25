import numpy as np


def bitonic_layer_loop(n):
    """Outer loop of a bitonic sort, which 
    iterates over the sublayers of a bitonic network"""
    layers = int(np.log2(n))
    for layer in range(layers):
        for s in range(layer+1):
            m = 1 << (layer - s)
            yield n, m, layer
            
def bitonic_swap_loop(n, m, layer):
    """Inner loop of a bitonic sort,
    which yields the elements to be swapped"""
    out = 0
    for i in range(0, n, m<<1):
        for j in range(m):
            ix = i + j
            a, b = ix, ix + m
            swap = (ix >> (layer + 1)) & 1
            yield a, b, out, swap
            out += 1

def bitonic_network(n):
    """Check the computation of a bitonic network, by printing
    the swapping layers, one permutation per line,
    and a divider after each complete layer block"""    
    for n,m,layer in bitonic_layer_loop(n):
        for a, b, out, swap in bitonic_swap_loop(n, m, layer):
            direction = "<" if swap else  ">"
            print(f"{a:>2}{direction}{b:<d}", end="\t")
        print()
        if m==1:
            print("-"*n*4)
            last_layer = layer


def pretty_bitonic_network(n):
    """Pretty print a bitonic network,
    to check the logic is correct"""
    layers = int(np.log2(n))
    # header
    for i in range(n):
        print(f" {i:<2d}", end="")
    print()

    for n,m,layer in bitonic_layer_loop(n):
        for a, b, out, swap in bitonic_swap_loop(n, m, layer):
            # this could be neater...
            for k in range(n):
                if a == k:                            
                    if swap:
                        print(" ╰─", end="")
                    else:
                        print(" ╭─", end="")
                elif b==k:
                    if swap:
                        print("─╮ ", end="")
                    else:
                        print("─╯ ", end="")
                elif a < k < b:
                    print("───", end="")
                else:
                    print(" │ ", end="")
            print()
    
