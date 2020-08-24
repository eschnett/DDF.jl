using Test

using Random
using SparseArrays

# Ignore a statement
macro DISABLED(expr)
    return quote end
end
# # Don't ignore a statement
# macro DISABLED(expr)
#     expr
# end

const Dmax = 5

# Random rationals
function Base.rand(rng::AbstractRNG,
                   ::Random.SamplerType{Rational{T}}) where {T}
    return Rational{T}(T(rand(rng, -1000:1000)) // 1000)
end

# Improved debug output for sparse matrices
Base.show(io::IO, A::SparseMatrixCSC) = show_sparse(io, A)
Base.show(io::IOContext, A::SparseMatrixCSC) = show_sparse(io, A)

include("test-defs.jl")
include("test-algorithms.jl")
# include("test-topologies.jl")
# include("test-funs.jl")
# include("test-ops.jl")
# include("test-geometries.jl")
include("test-zeroorone.jl")
include("test-sparseops.jl")
include("test-manifolds.jl")
include("test-funs.jl")
