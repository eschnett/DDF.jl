module DDF

using Reexport

# The order of these include statements matters
include("Defs.jl")
include("Algorithms.jl")
include("Topologies.jl")
include("Funs.jl")
include("Ops.jl")
include("Geometries.jl")

@reexport using .Algorithms
@reexport using .Defs
@reexport using .Funs
@reexport using .Geometries
@reexport using .Ops
@reexport using .Topologies

end
