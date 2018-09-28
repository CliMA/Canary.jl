```@meta
DocTestSetup = :(using Canary)
```

# Canary.jl
*An exploration of discontinuous Galerkin methods in Julia.*

## Installation

To install, run the following commands in the Julia REPL:

```julia
] add "https://github.com/climate-machine/Canary.jl"
```
and then run

```julia
using Canary
```
to load the package.

If you are using some of the MPI based functions at the REPL you will also need
to load MPI with something like

```julia
using MPI
MPI.Init()
MPI.finalize_atexit()
```

If you have problems building MPI.jl try explicitly providing the MPI compiler
wrappers through the `CC` and `FC` environment variables.  Something like the
following at your Bourne shell prompt:

```sh
export CC=mpicc
export FC=mpif90
```

Then launch Julia rebuild MPI.jl with

```julia
] build MPI
```

## Building Documentation

You may build the documentation locally by running

```sh

julia --color=yes --project=docs/ -e 'using Pkg; Pkg.instantiate()'

julia --color=yes --project=docs/ docs/make.jl

```

where first invocation of `julia` only needs to be run the first time.
