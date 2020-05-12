using Test

using SparseArrays



const Dmax = 5



# Random rationals
Base.rand(::Type{Rational{T}}) where {T} =
    Rational{T}(T(rand(Int16)) // 1000)
Base.rand(::Type{Rational{T}}, n::Int) where {T} =
    Rational{T}[rand(Rational{T}) for i in 1:n]



Base.show(io::IO, A::SparseMatrixCSC) = show_sparse(io, A)
Base.show(io::IOContext, A::SparseMatrixCSC) = show_sparse(io, A)



include("test-defs.jl")
include("test-manifolds.jl")
include("test-funs.jl")
include("test-geometries.jl")
