# generate examples
import Literate

EXAMPLEDIR = joinpath(@__DIR__, "..", "examples")
EXAMPLES = [joinpath(EXAMPLEDIR, f) for f in ("advection.jl",)]
GENERATEDDIR = joinpath(@__DIR__, "src", "examples", "generated")


mkpath(GENERATEDDIR)

for f in ("vtk.jl", "advection2d.png")
  cp(joinpath(EXAMPLEDIR, f), joinpath(GENERATEDDIR, f); force = true)
end

for input in EXAMPLES
  script = Literate.script(input, GENERATEDDIR)
  code = strip(read(script, String))
  mdpost(str) = replace(str, "@__CODE__" => code)
  Literate.markdown(input, GENERATEDDIR, postprocess = mdpost)
end
