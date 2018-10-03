# generate examples
import Literate


EXAMPLES = ("advection", "shallow_water")
EXAMPLEDIR = joinpath(@__DIR__, "..", "examples")
EXAMPLEFILES = [joinpath(EXAMPLEDIR, "$f.jl") for f in EXAMPLES]
GENERATEDDIR = joinpath(@__DIR__, "src", "examples", "generated")


mkpath(GENERATEDDIR)

for f in ("vtk.jl", "advection2d.png", "shallow_water2d.png")
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
