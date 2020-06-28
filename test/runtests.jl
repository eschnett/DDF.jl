using Test

using Random
using SparseArrays



const Dmax = 5



# Random rationals
Base.rand(rng::AbstractRNG, ::Random.SamplerType{Rational{T}}) where {T} =
    Rational{T}(T(rand(rng, -1000:1000)) // 1000)



# Improved debug output for sparse matrices
Base.show(io::IO, A::SparseMatrixCSC) = show_sparse(io, A)
Base.show(io::IOContext, A::SparseMatrixCSC) = show_sparse(io, A)



#TODO include("test-defs.jl")
#TODO include("test-forms.jl")
#TODO include("test-algorithms.jl")
#TODO include("test-topologies.jl")
#TODO include("test-funs.jl")
#TODO include("test-ops.jl")
include("test-geometries.jl")
