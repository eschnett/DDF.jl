module Funs

using DifferentialForms: Forms
using SparseArrays
using StaticArrays

using ..Defs
using ..Manifolds

export Fun
"""
Function (aka Cochain)
"""
struct Fun{D,P,R,S,T}         # <: AbstractVector{T}
    manifold::Manifold{D,S}
    values::Vector{T}

    function Fun{D,P,R,S,T}(manifold::Manifold{D,S},
                            values::AbstractVector{T}) where {D,P,R,S,T}
        D::Int
        @assert D >= 0
        P::PrimalDual
        R::Int
        @assert 0 <= R <= D
        fun = new{D,P,R,S,T}(manifold, values)
        @assert invariant(fun)
        return fun
    end
    function Fun{D,P,R,S}(manifold::Manifold{D,S},
                          values::AbstractVector{T}) where {D,P,R,S,T}
        return Fun{D,P,R,S,T}(manifold, values)
    end
    function Fun{D,P,R}(manifold::Manifold{D,S},
                        values::AbstractVector{T}) where {D,P,R,S,T}
        return Fun{D,P,R,S,T}(manifold, values)
    end
end

function Base.show(io::IO, fun::Fun{D,P,R,S,T}) where {D,P,R,S,T}
    println(io)
    println(io, "Fun{$D,$P,$R,$S,$T}(")
    println(io, "    manifold=$(fun.manifold.name)")
    println(io, "    values=$(fun.values)")
    return print(io, ")")
end

function Defs.invariant(fun::Fun{D,P,R,S,T})::Bool where {D,P,R,S,T}
    D::Int
    P::PrimalDual
    R::Int
    0 <= R <= D || return false
    invariant(fun.manifold) || return false
    length(fun.values) == nsimplices(fun.manifold, R) || return false
    return true
end

# Comparison

function Base.:(==)(f::F, g::F) where {F<:Fun}
    @assert f.manifold == g.manifold
    return f.values == g.values
end
function Base.:(<)(f::F, g::F) where {F<:Fun}
    @assert f.manifold == g.manifold
    return f.values < g.values
end
function Base.isequal(f::F, g::F) where {F<:Fun}
    @assert f.manifold == g.manifold
    return isequal(f.values, g.values)
end
function Base.hash(f::Fun, h::UInt)
    return hash(0xfc734743, hash(f.manifold, hash(f.values, h)))
end

# Random functions
function Base.rand(::Type{Fun{D,P,R,S,T}},
                   manifold::Manifold{D,S}) where {D,P,R,S,T}
    return Fun{D,P,R,S}(manifold, rand(T, nsimplices(manifold, R)))
end

# Functions are a collection

Base.IteratorEltype(::Type{<:Fun{<:Any,<:Any,<:Any,<:Any,T}}) where {T} = T
function Base.IteratorSize(::Type{<:Fun{<:Any,<:Any,<:Any,<:Any,T}}) where {T}
    Base.IteratorSize(Vector{T})
end
Base.eltype(::Type{<:Fun{<:Any,<:Any,<:Any,<:Any,T}}) where {T} = T
Base.eltype(f::Fun) = eltype(typeof(f))
Base.isempty(f::Fun) = isempty(f.values)
Base.iterate(f::Fun, state...) = iterate(f.values, state...)
Base.length(f::Fun) = length(f.values)

# function Base.map(op, f::Fun{D,P,R}, gs::Fun{D,P,R}...) where {D,P,R}
#     @assert all(f.manifold == g.manifold for g in gs)
#     return Fun{D,P,R}(f.manifold,
#                         map(op, f.values, (g.values for g in gs)...))
#     # r1 = map(op, f.values[1], (g.values[1] for g in gs)...))
#     # T = typeof(r1)
#     # Fun{D, P, R, S, T}(f.manifold, map(op, f.values, (g.values for g in gs)...))
# end
function Base.map(op, f::Fun{D,P,R,S}, gs::Fun{D,P,R,S}...) where {D,P,R,S}
    @assert all(g.manifold == f.manifold for g in gs)
    U = typeof(op(zero(eltype(f)), (zero(eltype(g)) for g in gs)...))
    Fun{D,P,R,S,U}(f.manifold, map(op, f.values, (g.values for g in gs)...))
end
function Base.reduce(op, f::Fun{D,P,R,S}, gs::Fun{D,P,R,S}...;
                     kw...) where {D,P,R,S}
    @assert all(g.manifold == f.manifold for g in gs)
    reduce(op, f.values, (g.values for g in gs)...; kw...)
end

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

function Base.zero(::Type{Fun{D,P,R,S,T}},
                   manifold::Manifold{D,S}) where {D,P,R,S,T}
    nelts = nsimplices(manifold, R)
    return Fun{D,P,R}(manifold, zeros(T, nelts))
end

function Forms.unit(::Type{Fun{D,P,R,S,T}}, manifold::Manifold{D,S},
                    n::Int) where {D,P,R,S,T}
    nelts = nsimplices(manifold, R)
    @assert 1 <= n <= nelts
    # return Fun{D,P,R,S}(manifold, sparsevec([n], [one(T)],nelts))
    return Fun{D,P,R,S}(manifold, T[i == n for i in 1:nelts])
end

function Base.:+(f::Fun{D,P,R,S}) where {D,P,R,S}
    return Fun{D,P,R,S}(f.manifold, +f.values)
end

function Base.:-(f::Fun{D,P,R,S}) where {D,P,R,S}
    return Fun{D,P,R,S}(f.manifold, -f.values)
end

function Base.:+(f::Fun{D,P,R,S}, g::Fun{D,P,R,S}) where {D,P,R,S}
    @assert f.manifold == g.manifold
    return Fun{D,P,R,S}(f.manifold, f.values + g.values)
end

function Base.:-(f::Fun{D,P,R,S}, g::Fun{D,P,R,S}) where {D,P,R,S}
    @assert f.manifold == g.manifold
    return Fun{D,P,R,S}(f.manifold, f.values - g.values)
end

function Base.:*(a::Number, f::Fun{D,P,R,S}) where {D,P,R,S}
    return Fun{D,P,R,S}(f.manifold, a * f.values)
end

function Base.:\(a::Number, f::Fun{D,P,R,S}) where {D,P,R,S}
    return Fun{D,P,R,S}(f.manifold, a \ f.values)
end

function Base.:*(f::Fun{D,P,R,S}, a::Number) where {D,P,R,S}
    return Fun{D,P,R,S}(f.manifold, f.values * a)
end

function Base.:/(f::Fun{D,P,R,S}, a::Number) where {D,P,R,S}
    return Fun{D,P,R,S}(f.manifold, f.values / a)
end

# Functions are a category

export id
function id(::Type{<:Fun{D,P,0,S,SVector{D,S}}},
            manifold::Manifold{D,S}) where {D,P,S}
    values = [SVector{D,S}(@view manifold.coords[n, :])
              for n in 1:nsimplices(manifold, 0)]
    return Fun{D,P,0,S}(manifold, values)
end

function Base.conj(f::Fun{D,P,R,S}) where {D,P,R,S}
    return Fun{D,P,R,S}(f.manifold, conj(f.values))
end

# composition?

end
