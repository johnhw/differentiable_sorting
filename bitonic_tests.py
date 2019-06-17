import numpy as np


def bitonic_network(n):
    """Check the computation of a bitonic network"""
    layers = int(np.log2(n))
    for layer in range(1, layers + 1):
        for sub in reversed(range(1, layer + 1)):
            for i in range(0, n, 2 ** sub):
                for j in range(2 ** (sub - 1)):
                    ix = i + j
                    a, b = ix, ix + (2 ** (sub - 1))
                    swap = "<" if (ix >> layer) & 1 else ">"
                    print(f"{a:>2}{swap}{b:<d}", end="\t")
            print()
        print("-" * n * 4)


def pretty_bitonic_network(n):
    """Pretty print a bitonic network,
    to check the logic is correct"""
    layers = int(np.log2(n))
    # header
    for i in range(n):
        print(f" {i:<2d}", end="")
    print()

    # layers
    width = 4
    grid = [["|"]*(n*width) for i in range(layers*layers)] 

    for layer in range(1, layers + 1):
        for sub in reversed(range(1, layer + 1)):
            for i in range(0, n, 2 ** sub):
                for j in range(2 ** (sub - 1)):
                    ix = i + j
                    a, b = ix, ix + (2 ** (sub - 1))
                    swap = (ix >> layer) & 1 
                    
                    # this could be neater...
                    for k in range(n):
                        if a == k:                            
                            if swap:
                                print(f" ╰─", end="")
                            else:
                                print(f" ╭─", end="")
                        elif b==k:
                            if swap:
                                print(f"─╮ ", end="")
                            else:
                                print(f"─╯ ", end="")
                        elif a < k < b:
                            print("───", end="")
                        else:
                            print(" │ ", end="")
                        
                    print()
