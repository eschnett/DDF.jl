module Ops

using LinearAlgebra
using SparseArrays

using ..Defs
using ..Funs
using ..Manifolds



# Operators

# TODO: Define these in Ops (or Funs?)
# TODO: Test them (similar to Funs)

# Drop non-structural zeros if we can do that without losing accuracy
const ExactType = Union{Integer, Rational}
compress(A) = A
compress(A::AbstractSparseMatrix{<:ExactType}) = dropzeros(A)
compress(A::AbstractSparseMatrix{Complex{<:ExactType}}) = dropzeros(A)
compress(A::Adjoint) = adjoint(compress(adjoint(A)))
compress(A::Transpose) = transpose(compress(transposes(value)))
# compress(A::Union{LowerTriangular, UpperTriangular}) =
#     typeof(A)(compress(A.data))

export Op
struct Op{D, R1, R2, T}         # <: AbstractMatrix{T}
    mf::DManifold{D}
    values::Union{AbstractMatrix{T}, UniformScaling{T}}
    # TODO: Check invariant

    function Op{D, R1, R2, T}(mf::DManifold{D},
                              values::Union{AbstractMatrix{T},
                                            UniformScaling{T}}
                              ) where {D, R1, R2, T}
        op = new{D, R1, R2, T}(mf, compress(values))
        @assert invariant(op)
        op
    end
    function Op{D, R1, R2}(mf::DManifold{D},
                           values::Union{AbstractMatrix{T}, UniformScaling{T}}
                           ) where {D, R1, R2, T}
        Op{D, R1, R2, T}(mf, values)
    end
end

function Defs.invariant(op::Op{D, R1, R2})::Bool where {D, R1, R2}
    D::Int
    @assert D >= 0
    R1::Int
    @assert 0 <= R1 <= D
    R2::Int
    @assert 0 <= R2 <= D
    if !(op.values isa UniformScaling)
        @assert size(op.values) == (size(R1, op.mf), size(R2, op.mf))
    end
    true
end

# Comparison

function Base.:(==)(A::M, B::M) where {M<:Op}
    @assert A.mf == B.mf
    A.values == B.values
end

# Operators are a collection

Base.iterate(A::Op, state...) = iterate(A.values, state...)
Base.iterate(A::Op, state...) = iterate(A.values, state...)
Base.IteratorSize(A::Op) = Base.IteratorSize(A.values)
Base.IteratorEltype(A::Op) = Base.IteratorEltype(A.values)
Base.isempty(A::Op) = isempty(A.values)
Base.length(A::Op) = length(A.values)
Base.eltype(A::Op) = eltype(A.values)

function Base.map(op, A::Op{D, R1, R2}, Bs::Op{D, R1, R2}...) where {D, R1, R2}
    @assert all(A.mf == B.mf for B in Bs)
    Fun{D, R}(A.mf, map(op, A.values, (B.values for B in Bs)...))
end

# Random operators
function Base.rand(::Type{Op{D, R1, R2, T}}, mf::DManifold{D}
                   ) where {D, R1, R2, T}
    m, n = size(R1, mf), size(R2, mf)
    p = clamp(4 / min(m, n), 0, 1)
    Op{D, R1, R2}(mf, sprand(T, m, n, p))
end

# Operators are an abstract matrix

Base.ndims(::Op) = 2
Base.size(A::Op) = size(A.values)
Base.size(A::Op, dims) = size(A.values, dims)
Base.axes(A::Op) = axes(A.values)
Base.axes(A::Op, dir) = axes(A.values, dir)
Base.eachindex(A::Op) = eachindex(A.values)
Base.IndexStyle(::Type{<:Op}) = IndexStyle(Vector)
Base.stride(A::Op, k) = stride(A.values, k)
Base.strides(A::Op) = strides(A.values)
Base.getindex(A::Op, inds...) = getindex(A.values, inds...)

# Operators are a vector space

function Base.zero(::Type{Op{D, R1, R2, T}}, mf::DManifold{D}
                   ) where {D, R1, R2, T}
    Op{D, R1, R2}(mf, sparse([], [], T[], size(R1, mf), size(R2, mf)))
end

function Defs.unit(::Type{Op{D, R1, R2, T}}, mf::DManifold{D}, m::Int, n::Int
                   ) where {D, R1, R2, T}
    @assert 1 <= m <= size(R1, mf)
    @assert 1 <= n <= size(R2, mf)
    Op{D, R1, R2}(mf, sparse([m], [n], [one(T)]))
end

function Base.:+(A::Op{D, R1, R2}) where {D, R1, R2}
    Op{D, R1, R2}(A.mf, +A.values)
end

function Base.:-(A::Op{D, R1, R2}) where {D, R1, R2}
    Op{D, R1, R2}(A.mf, -A.values)
end

function Base.:+(A::Op{D, R1, R2}, B::Op{D, R1, R2}) where {D, R1, R2}
    @assert A.mf == B.mf
    Op{D, R1, R2}(A.mf, A.values + B.values)
end

function Base.:-(A::Op{D, R1, R2}, B::Op{D, R1, R2}) where {D, R1, R2}
    @assert A.mf == B.mf
    Op{D, R1, R2}(A.mf, A.values - B.values)
end

function Base.:*(a::Number, A::Op{D, R1, R2}) where {D, R1, R2}
    Op{D, R1, R2}(A.mf, a * A.values)
end

function Base.:\(a::Number, A::Op{D, R1, R2}) where {D, R1, R2}
    Op{D, R1, R2}(A.mf, a \ A.values)
end

function Base.:*(A::Op{D, R1, R2}, a::Number) where {D, R1, R2}
    Op{D, R1, R2}(A.mf, A.values * a)
end

function Base.:/(A::Op{D, R1, R2}, a::Number) where {D, R1, R2}
    Op{D, R1, R2}(A.mf, A.values / a)
end

# Operators are a ring

function Base.one(::Type{Op{D, R, R}}, mf::DManifold{D}) where {D, R}
    Op{D, R, R}(mf, I)
end
function Base.one(::Type{Op{D, R, R, T}}, mf::DManifold{D}) where {D, R, T}
    Op{D, R, R}(mf, one(T)*I)
end

function Base.:*(A::Op{D, R1, R2}, B::Op{D, R2, R3}) where {D, R1, R2, R3}
    @assert A.mf == B.mf
    Op{D, R1, R3}(A.mf, A.values * B.values)
end

# # Operators are a division ring
# 
# function Base.inv(A::Op{D, R1, R2}) where {D, R1, R2}
#     Op{D, R2, R1}(A.mf, inv(A.values))
# end
# 
# function Base.:/(A::Op{D, R1, R2}, B::Op{D, R3, R2}) where {D, R1, R2, R3}
#     @assert A.mf == B.mf
#     Op{D, R1, R3}(A.mf, A.values / B.values)
# end
# 
# function Base.:\(A::Op{D, R2, R1}, B::Op{D, R2, R3}) where {D, R1, R2, R3}
#     @assert A.mf == B.mf
#     Op{D, R1, R3}(A.mf, A.values \ B.values)
# end

# There are adjoints

function Base.adjoint(A::Op{D, R1, R2}) where {D, R1, R2}
    Op{D, R2, R1}(A.mf, adjoint(A.values))
end

function Base.transpose(A::Op{D, R1, R2}) where {D, R1, R2}
    Op{D, R2, R1}(A.mf, transpose(A.values))
end

# TODO: shouldn't this be defined automatically by being a collection?
function Base.conj(A::Op{D, R1, R2}) where {D, R1, R2}
    Op{D, R1, R2}(A.mf, conj(A.values))
end

# Operators act on functions

function Base.:*(A::Op{D, R1, R2}, f::Fun{D, R2}) where {D, R1, R2}
    @assert A.mf == f.mf
    Fun{D, R1}(f.mf, A.values * f.values)
end

function Base.:\(A::Op{D, R1, R2}, f::Fun{D, R1}) where {D, R1, R2}
    @assert A.mf == f.mf
    # Note: \ converts rationals to Float64
    Fun{D, R2}(f.mf, A.values \ f.values)
end



# Boundary

export boundary
function boundary(::Val{R}, mf::DManifold{D}) where {R, D}
    @assert 0 < R <= D
    Op{D, R-1, R, Int8}(mf, mf.boundaries[R])
end

# Derivative

export deriv
function deriv(::Val{R}, mf::DManifold{D}) where {R, D}
    @assert 0 <= R < D
    boundary(Val(R+1), mf)'
end

end
