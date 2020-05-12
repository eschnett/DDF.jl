module DDF

using Reexport

# The order of include statements matters
include("Defs.jl")
include("Manifolds.jl")
include("Funs.jl")
include("Geometries.jl")

@reexport using .Defs
@reexport using .Manifolds
@reexport using .Funs
@reexport using .Geometries

end
