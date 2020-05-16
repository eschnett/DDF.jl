module DDF

using Reexport

# The order of include statements matters
include("Defs.jl")
include("Manifolds.jl")
include("Funs.jl")
include("Ops.jl")
include("Geometries.jl")

@reexport using .Defs
@reexport using .Manifolds
@reexport using .Funs
@reexport using .Ops
@reexport using .Geometries

end
