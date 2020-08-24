module DDF

using Reexport

# The order of these include statements matters
include("Defs.jl")
include("Algorithms.jl")
include("ZeroOrOne.jl")
include("SparseOps.jl")
include("Manifolds.jl")
include("ManifoldConstructors.jl")
include("Funs.jl")
include("Ops.jl")
# include("Topologies.jl")
# include("Geometries.jl")

@reexport using .Algorithms
@reexport using .Defs
@reexport using .Funs
@reexport using .ManifoldConstructors
@reexport using .Manifolds
@reexport using .Ops
@reexport using .SparseOps
@reexport using .ZeroOrOne
# @reexport using .Geometries
# @reexport using .Topologies

end
