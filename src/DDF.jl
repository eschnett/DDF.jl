module DDF

using Reexport

# The order of these include statements matters
include("Defs.jl")
include("Algorithms.jl")
# include("Topologies.jl")
# include("Funs.jl")
# include("Ops.jl")
# include("Geometries.jl")
include("ZeroOrOne.jl")
include("SparseOps.jl")
include("Manifolds.jl")
include("ManifoldConstructors.jl")

@reexport using .Algorithms
@reexport using .Defs
@reexport using .ManifoldConstructors
@reexport using .Manifolds
@reexport using .SparseOps
@reexport using .ZeroOrOne
# @reexport using .Funs
# @reexport using .Geometries
# @reexport using .Ops
# @reexport using .Topologies

end
