module Funs

using SparseArrays

using ..Defs
using ..Topologies



export PrimalDual, Pr, Dl
@enum PrimalDual::Bool Pr Dl

Base.:!(P::PrimalDual) = PrimalDual(!(Bool(P)))



export Fun
"""
Function (aka Cochain)
"""
struct Fun{D, P, R, T}         # <: AbstractVector{T}
    topo::Topology{D}
    values::AbstractVector{T}

    function Fun{D, P, R, T}(topo::Topology{D},
                             values::AbstractVector{T}) where {D, P, R, T}
        D::Int
        @assert D >= 0
        P::PrimalDual
        R::Int
        @assert 0 <= R <= D
        fun = new{D, P, R, T}(topo, values)
        @assert invariant(fun)
        fun
    end
    function Fun{D, P, R}(topo::Topology{D},
                          values::AbstractVector{T}) where {D, P, R, T}
        Fun{D, P, R, T}(topo, values)
    end
end

function Defs.invariant(fun::Fun{D, P, R, T})::Bool where {D, P, R, T}
    D::Int
    P::PrimalDual
    R::Int
    0 <= R <= D || return false
    invariant(fun.topo) || return false
    length(fun.values) == size(R, fun.topo) || return false
    return true
end

# Comparison

function Base.:(==)(f::F, g::F) where {F<:Fun}
    @assert f.topo == g.topo
    f.values == g.values
end

# Functions are a collection

Base.iterate(f::Fun, state...) = iterate(f.values, state...)
Base.IteratorSize(f::Fun) = Base.IteratorSize(f.values)
Base.IteratorEltype(f::Fun) = Base.IteratorEltype(f.values)
Base.isempty(f::Fun) = isempty(f.values)
Base.length(f::Fun) = length(f.values)
Base.eltype(f::Fun) = eltype(f.values)

function Base.map(op, f::Fun{D, P, R}, gs::Fun{D, P, R}...) where {D, P, R}
    @assert all(f.topo == g.topo for g in gs)
    Fun{D, P, R}(f.topo, map(op, f.values, (g.values for g in gs)...))
end

# Random functions
Base.rand(::Type{Fun{D, P, R, T}}, topo::Topology{D}) where {D, P, R, T} =
    Fun{D, P, R, T}(topo, rand(T, size(R, topo)))

# Functions are an abstract vector

Base.ndims(::Fun) = 1
Base.size(f::Fun) = size(f.values)
Base.size(f::Fun, dims) = size(f.values, dims)
Base.axes(f::Fun) = axes(f.values)
Base.axes(f::Fun, dir) = axes(f.values, dir)
Base.eachindex(f::Fun) = eachindex(f.values)
Base.IndexStyle(::Type{<:Fun}) = IndexStyle(Vector)
Base.stride(f::Fun, k) = stride(f.values, k)
Base.strides(f::Fun) = strides(f.values)
Base.getindex(f::Fun, inds...) = getindex(f.values, inds...)

# Functions are a vector space

function Base.zero(::Type{Fun{D, P, R, T}}, topo::Topology{D}) where {D, P, R, T}
    Fun{D, P, R}(topo, zeros(T, size(R, topo)))
end

function Defs.unit(::Type{Fun{D, P, R, T}}, topo::Topology{D}, n::Int
                   ) where {D, P, R, T}
    @assert 1 <= n <= size(R, topo)
    Fun{D, P, R}(topo, sparsevec([n], [one(T)]))
end

function Base.:+(f::Fun{D, P, R}) where {D, P, R}
    Fun{D, P, R}(f.topo, +f.values)
end

function Base.:-(f::Fun{D, P, R}) where {D, P, R}
    Fun{D, P, R}(f.topo, -f.values)
end

function Base.:+(f::Fun{D, P, R}, g::Fun{D, P, R}) where {D, P, R}
    @assert f.topo == g.topo
    Fun{D, P, R}(f.topo, f.values + g.values)
end

function Base.:-(f::Fun{D, P, R}, g::Fun{D, P, R}) where {D, P, R}
    @assert f.topo == g.topo
    Fun{D, P, R}(f.topo, f.values - g.values)
end

function Base.:*(a::Number, f::Fun{D, P, R}) where {D, P, R}
    Fun{D, P, R}(f.topo, a * f.values)
end

function Base.:\(a::Number, f::Fun{D, P, R}) where {D, P, R}
    Fun{D, P, R}(f.topo, a \ f.values)
end

function Base.:*(f::Fun{D, P, R}, a::Number) where {D, P, R}
    Fun{D, P, R}(f.topo, f.values * a)
end

function Base.:/(f::Fun{D, P, R}, a::Number) where {D, P, R}
    Fun{D, P, R}(f.topo, f.values / a)
end

# Functions have a pointwise product

function Base.zeros(::Type{Fun{D, P, R, T}}, topo::Topology{D}
                    ) where {D, P, R, T}
    Fun{D, P, R}(topo, zeros(T, size(R, topo)))
end

function Base.ones(::Type{Fun{D, P, R, T}}, topo::Topology{D}) where {D, P, R, T}
    Fun{D, P, R}(topo, ones(T, size(R, topo)))
end

# TODO: shouldn't this be defined automatically by being a collection?
function Base.conj(f::Fun{D, P, R}) where {D, P, R}
    Fun{D, P, R}(f.topo, conj(f.values))
end

# Functions are a category

export id
function id(::Type{Fun{D, P, R, T}}, topo::Topology{D}) where {D, P, R, T}
    Fun{D, P, R}(topo, [T(i) for i in 1:size(R, topo)])
end

# composition?

end
