Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625

using Documenter, Canary

# Generate examples
include("generate.jl")

GENERATEDEXAMPLES = [joinpath("examples", "generated", f) for f in ("advection.md",)]

# Build documentation.
makedocs(
         sitename = "Canary.jl",
         doctest = false,
         strict = false,
         pages = Any[
                     "Home" => "index.md",
                     "manual/dg_intro.md",
                     "Manual" => [
                                  "manual/mesh.md",
                                  "manual/metric.md",
                                  "manual/operators.md",
                                 ],
                     "Examples" => GENERATEDEXAMPLES,
                     "API Reference" => [
                                         "reference/mesh.md",
                                         "reference/metric.md",
                                         "reference/operators.md",
                                        ]
                    ],
         html_prettyurls = haskey(ENV, "HAS_JOSH_K_SEAL_OF_APPROVAL") # disable for local builds
        )

# Deploy built documentation from Travis.
deploydocs(
           repo = "github.com/climate-machine/Canary.jl.git",
           deps = Deps.pip("mkdocs", "pygments", "python-markdown-math"),
           julia = "1.0",
          )
