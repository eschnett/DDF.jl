module Funs

using SparseArrays

using ..Defs
using ..Manifolds



export PrimalDual, Pr, Dl
@enum PrimalDual::Bool Pr Dl

Base.:!(P::PrimalDual) = PrimalDual(!(Bool(P)))



export Fun
"""
Function (aka Cochain)
"""
struct Fun{D, P, R, T}         # <: AbstractVector{T}
    mf::DManifold{D}
    values::AbstractVector{T}

    function Fun{D, P, R, T}(mf::DManifold{D},
                             values::AbstractVector{T}) where {D, P, R, T}
        D::Int
        @assert D >= 0
        P::PrimalDual
        R::Int
        @assert 0 <= R <= D
        fun = new{D, P, R, T}(mf, values)
        @assert invariant(fun)
        fun
    end
    function Fun{D, P, R}(mf::DManifold{D},
                          values::AbstractVector{T}) where {D, P, R, T}
        Fun{D, P, R, T}(mf, values)
    end
end

function Defs.invariant(fun::Fun{D, P, R, T})::Bool where {D, P, R, T}
    D::Int
    P::PrimalDual
    R::Int
    0 <= R <= D || return false
    invariant(fun.mf) || return false
    length(fun.values) == size(R, fun.mf) || return false
    return true
end

# Comparison

function Base.:(==)(f::F, g::F) where {F<:Fun}
    @assert f.mf == g.mf
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
    @assert all(f.mf == g.mf for g in gs)
    Fun{D, P, R}(f.mf, map(op, f.values, (g.values for g in gs)...))
end

# Random functions
Base.rand(::Type{Fun{D, P, R, T}}, mf::DManifold{D}) where {D, P, R, T} =
    Fun{D, P, R, T}(mf, rand(T, size(R, mf)))

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

function Base.zero(::Type{Fun{D, P, R, T}}, mf::DManifold{D}) where {D, P, R, T}
    Fun{D, P, R}(mf, zeros(T, size(R, mf)))
end

function Defs.unit(::Type{Fun{D, P, R, T}}, mf::DManifold{D}, n::Int
                   ) where {D, P, R, T}
    @assert 1 <= n <= size(R, mf)
    Fun{D, P, R}(mf, sparsevec([n], [one(T)]))
end

function Base.:+(f::Fun{D, P, R}) where {D, P, R}
    Fun{D, P, R}(f.mf, +f.values)
end

function Base.:-(f::Fun{D, P, R}) where {D, P, R}
    Fun{D, P, R}(f.mf, -f.values)
end

function Base.:+(f::Fun{D, P, R}, g::Fun{D, P, R}) where {D, P, R}
    @assert f.mf == g.mf
    Fun{D, P, R}(f.mf, f.values + g.values)
end

function Base.:-(f::Fun{D, P, R}, g::Fun{D, P, R}) where {D, P, R}
    @assert f.mf == g.mf
    Fun{D, P, R}(f.mf, f.values - g.values)
end

function Base.:*(a::Number, f::Fun{D, P, R}) where {D, P, R}
    Fun{D, P, R}(f.mf, a * f.values)
end

function Base.:\(a::Number, f::Fun{D, P, R}) where {D, P, R}
    Fun{D, P, R}(f.mf, a \ f.values)
end

function Base.:*(f::Fun{D, P, R}, a::Number) where {D, P, R}
    Fun{D, P, R}(f.mf, f.values * a)
end

function Base.:/(f::Fun{D, P, R}, a::Number) where {D, P, R}
    Fun{D, P, R}(f.mf, f.values / a)
end

# Functions have a pointwise product

function Base.zeros(::Type{Fun{D, P, R, T}}, mf::DManifold{D}
                    ) where {D, P, R, T}
    Fun{D, P, R}(mf, zeros(T, size(R, mf)))
end

function Base.ones(::Type{Fun{D, P, R, T}}, mf::DManifold{D}) where {D, P, R, T}
    Fun{D, P, R}(mf, ones(T, size(R, mf)))
end

# TODO: shouldn't this be defined automatically by being a collection?
function Base.conj(f::Fun{D, P, R}) where {D, P, R}
    Fun{D, P, R}(f.mf, conj(f.values))
end

# Functions are a category

export id
function id(::Type{Fun{D, P, R, T}}, mf::DManifold{D}) where {D, P, R, T}
    Fun{D, P, R}(mf, [T(i) for i in 1:size(R, mf)])
end

# composition?

end
