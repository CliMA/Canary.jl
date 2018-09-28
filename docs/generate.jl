# generate examples
import Literate

EXAMPLEDIR = joinpath(@__DIR__, "..", "examples")
EXAMPLES = [joinpath(EXAMPLEDIR, f) for f in ("advection.jl",)]
GENERATEDDIR = joinpath(@__DIR__, "src", "examples", "generated")

mkpath(GENERATEDDIR)
cp(joinpath(EXAMPLEDIR, "vtk.jl"), joinpath(GENERATEDDIR, "vtk.jl");
   force = true)

for input in EXAMPLES
  script = Literate.script(input, GENERATEDDIR)
  code = strip(read(script, String))
  mdpost(str) = replace(str, "@__CODE__" => code)
  Literate.markdown(input, GENERATEDDIR, postprocess = mdpost)
end
