"""
    linearpartition(n, p, np)

Partition the range `1:n` into `np` pieces and return the `p`th piece as a
range.

This will provide an equal partition when `n` is divisible by `np` and otherwise
the ranges will have lengths of either `floor(Int, n/np)` or `ceil(Int, n/np)`.
"""
linearpartition(n, p, np) = range(div((p-1)*n, np) + 1, stop=div(p*n, np))
