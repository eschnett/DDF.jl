module Ops

using LinearAlgebra
using SparseArrays

using ..Defs
using ..Funs
using ..Topologies

# Operators

# TODO: Define these in Ops (or Funs?)
# TODO: Test them (similar to Funs)

# Drop non-structural zeros if we can do that without losing accuracy
const ExactType = Union{Integer,Rational}
compress(A) = A
compress(A::AbstractSparseMatrix{<:ExactType}) = dropzeros(A)
compress(A::AbstractSparseMatrix{Complex{<:ExactType}}) = dropzeros(A)
compress(A::Adjoint) = adjoint(compress(adjoint(A)))
compress(A::Transpose) = transpose(compress(transpose(A)))
# compress(A::Union{LowerTriangular, UpperTriangular}) =
#     typeof(A)(compress(A.data))

export Op
struct Op{D,P1,R1,P2,R2,T} # <: AbstractMatrix{T}
    topo::Topology{D}
    values::Union{AbstractMatrix{T},UniformScaling{T}}
    # TODO: Check invariant

    function Op{D,P1,R1,P2,R2,T}(topo::Topology{D},
                                 values::Union{AbstractMatrix{T},
                                               UniformScaling{T}}) where {D,P1,
                                                                          R1,P2,
                                                                          R2,T}
        op = new{D,P1,R1,P2,R2,T}(topo, compress(values))
        @assert invariant(op)
        return op
    end
    function Op{D,P1,R1,P2,R2}(topo::Topology{D},
                               values::Union{AbstractMatrix{T},
                                             UniformScaling{T}}) where {D,P1,R1,
                                                                        P2,R2,T}
        return Op{D,P1,R1,P2,R2,T}(topo, values)
    end
end

function Base.show(io::IO, op::Op{D,P1,R1,P2,R2,T}) where {D,P1,R1,P2,R2,T}
    println(io)
    println(io, "Op{$D,$P1,$R1,$P2,$R2,$T}(")
    println(io, "    topo=$(op.topo.name)")
    println(io, "    values=$(op.values)")
    return print(io, ")")
end

function Defs.invariant(op::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    D::Int
    @assert D >= 0
    P1::PrimalDual
    R1::Int
    @assert 0 <= R1 <= D
    P2::PrimalDual
    R2::Int
    @assert 0 <= R2 <= D
    if !(op.values isa UniformScaling)
        @assert size(op.values) == (size(R1, op.topo), size(R2, op.topo))
    end
    return true
end

# Comparison

function Base.:(==)(A::M, B::M) where {M<:Op}
    @assert A.topo == B.topo
    return A.values == B.values
end

# Operators are a collection

Base.iterate(A::Op, state...) = iterate(A.values, state...)
Base.IteratorSize(A::Op) = Base.IteratorSize(A.values)
Base.IteratorEltype(A::Op) = Base.IteratorEltype(A.values)
Base.isempty(A::Op) = isempty(A.values)
Base.length(A::Op) = length(A.values)
Base.eltype(A::Op) = eltype(A.values)

function Base.map(op, A::Op{D,P1,R1,P2,R2},
                  Bs::Op{D,P1,R1,P2,R2}...) where {D,P1,R1,P2,R2}
    @assert all(A.topo == B.topo for B in Bs)
    return Fun{D,R}(A.topo, map(op, A.values, (B.values for B in Bs)...))
end

# Random operators
function Base.rand(::Type{Op{D,P1,R1,P2,R2,T}},
                   topo::Topology{D}) where {D,P1,R1,P2,R2,T}
    m, n = size(R1, topo), size(R2, topo)
    p = clamp(4 / min(m, n), 0, 1)
    return Op{D,P1,R1,P2,R2}(topo, sprand(T, m, n, p))
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

function Base.zero(::Type{Op{D,P1,R1,P2,R2,T}},
                   topo::Topology{D}) where {D,P1,R1,P2,R2,T}
    return Op{D,P1,R1,P2,R2}(topo, spzeros(T, size(R1, topo), size(R2, topo)))
end

function Defs.unit(::Type{Op{D,P1,R1,P2,R2,T}}, topo::Topology{D}, m::Int,
                   n::Int) where {D,P1,R1,P2,R2,T}
    @assert 1 <= m <= size(R1, topo)
    @assert 1 <= n <= size(R2, topo)
    return Op{D,P1,R1,P2,R2}(topo, sparse([m], [n], [one(T)]))
end

function Base.:+(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.topo, +A.values)
end

function Base.:-(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.topo, -A.values)
end

function Base.:+(A::Op{D,P1,R1,P2,R2},
                 B::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    @assert A.topo == B.topo
    return Op{D,P1,R1,P2,R2}(A.topo, A.values + B.values)
end

function Base.:-(A::Op{D,P1,R1,P2,R2},
                 B::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    @assert A.topo == B.topo
    return Op{D,P1,R1,P2,R2}(A.topo, A.values - B.values)
end

function Base.:*(a::Number, A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.topo, a * A.values)
end

function Base.:\(a::Number, A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.topo, a \ A.values)
end

function Base.:*(A::Op{D,P1,R1,P2,R2}, a::Number) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.topo, A.values * a)
end

function Base.:/(A::Op{D,P1,R1,P2,R2}, a::Number) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.topo, A.values / a)
end

# Operators are a ring

function Base.one(::Type{Op{D,R,P,R,P}}, topo::Topology{D}) where {D,R,P}
    return Op{D,R,P,R,P}(topo, I)
end
function Base.one(::Type{Op{D,R,P,R,P,T}}, topo::Topology{D}) where {D,R,P,T}
    return Op{D,R,P,R,P}(topo, one(T) * I)
end

function Base.:*(A::Op{D,P1,R1,P2,R2},
                 B::Op{D,P2,R2,P3,R3}) where {D,P1,R1,P2,R2,P3,R3}
    @assert A.topo == B.topo
    return Op{D,P1,R1,P3,R3}(A.topo, A.values * B.values)
end

# Operators are a groupoid ("division ring")

# Note: This works only for invertible operators, and only for some
# matrix representations
function Base.inv(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    @assert R1 == R2
    @assert size(R1, A.topo) == size(R2, A.topo)
    return Op{D,P2,R2,P1,R1}(A.topo, inv(A.values))
end

function Base.:/(A::Op{D,P1,R1,P2,R2},
                 B::Op{D,P3,R3,P2,R2}) where {D,P1,R1,P2,R2,P3,R3}
    @assert A.topo == B.topo
    return Op{D,P1,R1,P3,R3}(A.topo, A.values / B.values)
end

function Base.:\(A::Op{D,P2,R2,P1,R1},
                 B::Op{D,P2,R2,P3,R3}) where {D,P1,R1,P2,R2,P3,R3}
    @assert A.topo == B.topo
    return Op{D,P1,R1,P3,R3}(A.topo, A.values \ B.values)
end

# There are adjoints

function Base.adjoint(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P2,R2,P1,R1}(A.topo, adjoint(A.values))
end

function Base.transpose(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P2,R2,P1,R1}(A.topo, transpose(A.values))
end

# TODO: shouldn't this be defined automatically by being a collection?
function Base.conj(A::Op{D,P1,R1,P2,R2}) where {D,P1,R1,P2,R2}
    return Op{D,P1,R1,P2,R2}(A.topo, conj(A.values))
end

# Operators act on functions

function Base.:*(A::Op{D,P1,R1,P2,R2}, f::Fun{D,P2,R2}) where {D,P1,R1,P2,R2}
    @assert A.topo == f.topo
    return Fun{D,P1,R1}(f.topo, A.values * f.values)
end

function Base.:\(A::Op{D,P1,R1,P2,R2}, f::Fun{D,P1,R1}) where {D,P1,R1,P2,R2}
    @assert A.topo == f.topo
    # Note: \ converts rationals to Float64
    return Fun{D,P2,R2}(f.topo, A.values \ f.values)
end

# Boundary

export boundary
function boundary(::Val{Pr}, ::Val{R}, topo::Topology{D}) where {R,D}
    @assert 0 < R <= D
    return Op{D,Pr,R - 1,Pr,R}(topo, topo.boundaries[R])
end
function boundary(::Val{Dl}, ::Val{R}, topo::Topology{D}) where {R,D}
    @assert 0 <= R < D
    return Op{D,Dl,R + 1,Dl,R}(topo, topo.boundaries[R + 1]')
end

export coboundary
function coboundary(::Val{Pr}, ::Val{R}, topo::Topology{D}) where {R,D}
    @assert 0 <= R < D
    return adjoint(boundary(Val(Pr), Val(R + 1), topo))::Op{D,Pr,R + 1,Pr,R}
end
function coboundary(::Val{Dl}, ::Val{R}, topo::Topology{D}) where {R,D}
    @assert 0 < R <= D
    return adjoint(boundary(Val(Dl), Val(R - 1), topo))::Op{D,Dl,R - 1,Dl,R}
end

# Derivative

export deriv
function deriv(::Val{Pr}, ::Val{R}, topo::Topology{D}) where {R,D}
    @assert 0 <= R < D
    return coboundary(Val(Pr), Val(R), topo)::Op{D,Pr,R + 1,Pr,R}
end
function deriv(::Val{Dl}, ::Val{R}, topo::Topology{D}) where {R,D}
    @assert 0 < R <= D
    return coboundary(Val(Dl), Val(R), topo)::Op{D,Dl,R - 1,Dl,R}
end

end
