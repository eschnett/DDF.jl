module Ops

using DifferentialForms: Forms
using FillArrays
using LinearAlgebra
using SparseArrays

using ..Defs
using ..Funs
using ..Manifolds
using ..SparseOps

# TODO: Add type `S` from `Manifold{D,S}`?
export Op
"""
Operator acting on `Fun` functions
"""
struct Op{D,P1,R1,P2,R2,T} # <: AbstractMatrix{T}
    manifold::Manifold{D}
    values::AbstractMatrix{T}

    function Op{D,P1,R1,P2,R2,T}(manifold::Manifold{D},
                                 values::AbstractMatrix{T}) where {D,P1,R1,P2,
                                                                   R2,T}
        op = new{D,P1,R1,P2,R2,T}(manifold, dnz(values))
        #TODO @assert invariant(op)
        return op
    end
    function Op{D,P1,R1,P2,R2}(manifold::Manifold{D},
                               values::AbstractMatrix{T}) where {D,P1,R1,P2,R2,
                                                                 T}
        return Op{D,P1,R1,P2,R2,T}(manifold, values)
    end
end

# Drop non-structural zeros if we can do that without losing accuracy
const ExactTypes = Union{Union{Integer,Rational},
                         Complex{<:Union{Integer,Rational}}}
const FloatTypes = Union{AbstractFloat,Complex{<:AbstractFloat}}
dnz(A) = A
function dnz(A::AbstractSparseMatrix{T}) where {T<:ExactTypes}
    return dropzeros(A)::AbstractSparseMatrix{T}
end
function dnz(A::AbstractSparseMatrix{T}) where {T<:FloatTypes}
    return dropzeros!(chop.(A))::AbstractSparseMatrix{T}
end
dnz(A::Adjoint) = adjoint(dnz(adjoint(A)))
dnz(A::Transpose) = transpose(dnz(transpose(A)))
# dnz(A::Union{LowerTriangular, UpperTriangular}) = typeof(A)(dnz(A.data))

chop(f::ExactTypes) = f
chop(f::FloatTypes) = ifelse(abs2(f) < eps34sq(typeof(f)), zero(f), f)
function chop(f::Complex{T}) where {T<:AbstractFloat}
    return Complex{T}(chop(real(f)), chop(imag(f)))
end

eps34sq(::Type{T}) where {T} = sqrt(eps(T)^3)::T
eps34(::Type{T}) where {T} = sqrt(eps34(T))::T

function Base.show(io::IO, op::Op{D,P1,R1,P2,R2,T}) where {D,P1,R1,P2,R2,T}
    println(io)
    println(io, "Op{$D,$P1,$R1,$P2,$R2,$T}(")
    println(io, "    manifold=$(op.manifold.name)")
    println(io, "    values=$(op.values)")
    return print(io, ")")
end

function Defs.invariant(op::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    D::Int
    @assert D ≥ 0
    P1::PrimalDual
    R1::Int
    @assert 0 ≤ R1 ≤ D
    P2::PrimalDual
    R2::Int
    @assert 0 ≤ R2 ≤ D
    @assert size(op.values) ==
            (nsimplices(op.manifold, R1), nsimplices(op.manifold, R2))
    return true
end

# Comparison

function Base.:(==)(A::M, B::M) where {M<:Op}
    @assert A.manifold ≡ B.manifold
    A ≡ B && return true
    return A.values == B.values
end
function Base.:(<)(A::M, B::M) where {M<:Op}
    @assert A.manifold ≡ B.manifold
    return A.values < B.values
end
function Base.isequal(A::M, B::M) where {M<:Op}
    @assert A.manifold ≡ B.manifold
    return isequal(A.values, B.values)
end
function Base.hash(A::Op, h::UInt)
    return hash(0xfc734743, hash(A.manifold, hash(A.values, h)))
end

# Random operators
function Base.rand(::Type{Op{D,P1,R1,P2,R2,T}},
                   manifold::Manifold{D}) where {D,P1,R1,P2,R2,T}
    nrows, ncols = nsimplices(manifold, R1), nsimplices(manifold, R2)
    p = clamp(4 / min(nrows, ncols), 0, 1)
    return Op{D,P1,R1,P2,R2}(manifold, sprand(T, nrows, ncols, p))
end

# Operators are a collection

Base.eltype(::Type{<:Op{<:Any,<:Any,<:Any,<:Any,<:Any,T}}) where {T} = T
Base.eltype(A::Op) = eltype(typeof(A))
Base.isempty(A::Op) = isempty(A.values)
Base.iterate(A::Op, state...) = iterate(A.values, state...)
Base.length(A::Op) = length(A.values)

# function Base.map(op, A::Op{D,P1,R1,P2,R2},
#                   Bs::Op{D,P1,R1,P2,R2}...) where {D,P1,R1,P2,R2}
#     @assert all(A.manifold ≡ B.manifold for B ∈ Bs)
#     return Fun{D,R}(A.manifold, map(op, A.values, (B.values for B ∈ Bs)...))
# end
function Base.map(op, A::Op{D,P1,R1,P2,R2},
                  Bs::Op{D,P1,R1,P2,R2}...) where {D,P1,R1,P2,R2}
    @assert all(B.manifold ≡ A.manifold for B in Bs)
    U = typeof(op(zero(eltype(A)), (zero(eltype(B)) for B in Bs)...))
    return Op{D,P1,R1,P2,R2,U}(A.manifold,
                               map(op, A.values, (B.values for B in Bs)...))
end
function Base.reduce(op, A::Op{D,P1,R1,P2,R2}, Bs::Op{D,P1,R1,P2,R2}...;
                     kw...) where {D,P1,R1,P2,R2}
    @assert all(B.manifold ≡ A.manifold for B in Bs)
    return reduce(op, A.values, (B.values for B in Bs)...; kw...)
end

# Operators are an abstract matrix

Base.IndexStyle(::Type{<:Op}) = IndexStyle(Vector)
Base.axes(A::Op) = axes(A.values)
Base.axes(A::Op, dir) = axes(A.values, dir)
Base.eachindex(A::Op) = eachindex(A.values)
Base.getindex(A::Op, inds...) = getindex(A.values, inds...)
Base.ndims(::Op) = 2
Base.size(A::Op) = size(A.values)
Base.size(A::Op, dims) = size(A.values, dims)
Base.stride(A::Op, k) = stride(A.values, k)
Base.strides(A::Op) = strides(A.values)

# Operators are a vector space

function Base.zero(::Type{Op{D,P1,R1,P2,R2,T}},
                   manifold::Manifold{D}) where {D,P1,R1,P2,R2,T}
    nrows = nsimplices(manifold, R1)
    ncols = nsimplices(manifold, R2)
    return Op{D,P1,R1,P2,R2}(manifold, spzeros(T, nrows, ncols))
end
function Base.zero(::Type{Op{D,P1,R1,P2,R2}},
                   manifold::Manifold{D}) where {D,P1,R1,P2,R2}
    return zero(Op{D,P1,R1,P2,R2,Bool}, manifold)
end
Base.zero(A::Op) = zero(typeof(A), A.manifold)
Base.iszero(A::Op) = iszero(A.values)

LinearAlgebra.norm(A::Op) = norm(A.values)
LinearAlgebra.norm(A::Op, p::Real) = norm(A.values, p)

function Forms.unit(::Type{Op{D,P1,R1,P2,R2,T}}, manifold::Manifold{D},
                    row::Int, col::Int) where {D,P1,R1,P2,R2,T}
    nrows = nsimplices(manifold, R1)
    ncols = nsimplices(manifold, R2)
    @assert 1 ≤ row ≤ nrows
    @assert 1 ≤ col ≤ ncols
    return Op{D,P1,R1,P2,R2}(manifold,
                             sparse([row], [col], [one(T)], nrows, ncols))
end

function Base.:+(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.manifold, +A.values)
end

function Base.:-(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.manifold, -A.values)
end

function Base.:+(A::Op{D,P1,R1,P2,R2},
                 B::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    @assert A.manifold ≡ B.manifold
    return Op{D,P1,R1,P2,R2}(A.manifold, A.values + B.values)
end

function Base.:-(A::Op{D,P1,R1,P2,R2},
                 B::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    @assert A.manifold ≡ B.manifold
    return Op{D,P1,R1,P2,R2}(A.manifold, A.values - B.values)
end

function Base.:*(a::Number, A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.manifold, a * A.values)
end

function Base.:\(a::Number, A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.manifold, a \ A.values)
end

function Base.:*(A::Op{D,P1,R1,P2,R2}, a::Number) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.manifold, A.values * a)
end

function Base.:/(A::Op{D,P1,R1,P2,R2}, a::Number) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.manifold, A.values / a)
end

# Operators are a ring

function Base.one(::Type{Op{D,P,R,P,R,T}},
                  manifold::Manifold{D}) where {D,P,R,T}
    nrows = nsimplices(manifold, R)
    return Op{D,P,R,P,R}(manifold, Diagonal(Fill(one(T), (nrows, nrows))))
end
function Base.one(::Type{Op{D,P,R,P,R}}, manifold::Manifold{D}) where {D,P,R}
    return one(Op{D,P,R,P,R,Bool}, manifold)
end
Base.one(A::Op) = one(typeof(A), A.manifold)
Base.isone(A::Op{D,P,R,P,R}) where {D,P,R} = A.values == I
Base.isone(A::Op) = false

function Base.:*(A::Op{D,P1,R1,P2,R2},
                 B::Op{D,P2,R2,P3,R3}) where {D,P1,R1,P2,R2,P3,R3}
    @assert A.manifold ≡ B.manifold
    return Op{D,P1,R1,P3,R3}(A.manifold, A.values * B.values)
end

# Operators are a groupoid ("division ring")

# Note: This works only for invertible operators, and only for some
# matrix representations
function Base.inv(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    @assert R1 == R2
    @assert size(R1, A.manifold) == size(R2, A.manifold)
    return Op{D,P2,R2,P1,R1}(A.manifold, inv(A.values))
end

function Base.:/(A::Op{D,P1,R1,P2,R2},
                 B::Op{D,P3,R3,P2,R2}) where {D,P1,R1,P2,R2,P3,R3}
    @assert A.manifold ≡ B.manifold
    return Op{D,P1,R1,P3,R3}(A.manifold, A.values / B.values)
end

function Base.:\(A::Op{D,P2,R2,P1,R1},
                 B::Op{D,P2,R2,P3,R3}) where {D,P1,R1,P2,R2,P3,R3}
    @assert A.manifold ≡ B.manifold
    return Op{D,P1,R1,P3,R3}(A.manifold, A.values \ B.values)
end

# There are adjoints

function Base.adjoint(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P2,R2,P1,R1}(A.manifold, adjoint(A.values))
end

function Base.transpose(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P2,R2,P1,R1}(A.manifold, transpose(A.values))
end

# TODO: shouldn't this be defined automatically by being a collection?
function Base.conj(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.manifold, conj(A.values))
end

# Operators act on functions

function Base.:*(A::Op{D,P1,R1,P2,R2}, f::Fun{D,P2,R2}) where {D,P1,R1,P2,R2}
    @assert A.manifold ≡ f.manifold
    return Fun{D,P1,R1}(f.manifold, IDVector{R1}(A.values * f.values.vec))
end

function Base.:\(A::Op{D,P1,R1,P2,R2}, f::Fun{D,P1,R1}) where {D,P1,R1,P2,R2}
    @assert A.manifold ≡ f.manifold
    # Note: \ converts rationals to Float64
    return Fun{D,P2,R2}(f.manifold, IDVector{R2}(A.values \ f.values.vec))
end

end
