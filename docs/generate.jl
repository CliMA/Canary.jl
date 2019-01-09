# generate examples
import Literate

GENERATEDEXAMPLES = String[]
for d = 1:3

  if d == 1
    EXAMPLES = ("LDG1d", "burger1d", "swe1d")
    EXTRA = ("vtk.jl",)
  elseif d == 2
    EXAMPLES = ("LDG2d", "swe2d", "nse2d")
    EXTRA = ("vtk.jl",)
  elseif d == 3
    EXAMPLES = ("LDG3d", "nse3d")
    EXTRA = ("vtk.jl",)
  end
  EXAMPLEDIR = joinpath(@__DIR__, "..", "examples", "$(d)d_kernels")
  EXAMPLEFILES = [joinpath(EXAMPLEDIR, "$f.jl") for f in EXAMPLES]
  GENERATEDDIR = joinpath(@__DIR__, "src", "examples", "generated",
                          "$(d)d_kernels")

  mkpath(GENERATEDDIR)

  for f in EXTRA
    cp(joinpath(EXAMPLEDIR, f), joinpath(GENERATEDDIR, f); force = true)
  end

  for input in EXAMPLEFILES
    script = Literate.script(input, GENERATEDDIR)
    code = strip(read(script, String))
    mdpost(str) = replace(str, "@__CODE__" => code)
    Literate.markdown(input, GENERATEDDIR, postprocess = mdpost)
  end

  # remove any .vtu files in the generated dir (should not be deployed)
  cd(GENERATEDDIR) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
  end

  for f in EXAMPLES
    md = joinpath("examples", "generated", "$(d)d_kernels", "$f.md")

    append!(GENERATEDEXAMPLES, (md,))
  end
end
