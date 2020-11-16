module Funs

using DifferentialForms: Forms, unit
using LinearAlgebra
using SparseArrays
using StaticArrays

using ..Defs
using ..Manifolds
using ..SparseOps

export Fun
"""
Function (aka Cochain)
"""
struct Fun{D,P,R,C,S,T}         # <: AbstractVector{T}
    manifold::Manifold{D,C,S}
    values::IDVector{R,T}

    function Fun{D,P,R,C,S,T}(manifold::Manifold{D,C,S},
                              values::IDVector{R,T}) where {D,P,R,C,S,T}
        D::Int
        @assert D ≥ 0
        P::PrimalDual
        R::Int
        @assert 0 ≤ R ≤ D
        fun = new{D,P,R,C,S,T}(manifold, values)
        #TODO @assert invariant(fun)
        return fun
    end
end

function Fun{D,P,R,C,S}(manifold::Manifold{D,C,S},
                        values::IDVector{R,T}) where {D,P,R,C,S,T}
    return Fun{D,P,R,C,S,T}(manifold, values)
end
function Fun{D,P,R}(manifold::Manifold{D,C,S},
                    values::IDVector{R,T}) where {D,P,R,C,S,T}
    return Fun{D,P,R,C,S,T}(manifold, values)
end

function Base.show(io::IO, fun::Fun{D,P,R,C,S,T}) where {D,P,R,C,S,T}
    println(io)
    println(io, "Fun{$D,$P,$R,$S,$T}(")
    println(io, "    manifold=$(fun.manifold.name)")
    println(io, "    values=$(fun.values)")
    return print(io, ")")
end

function Defs.invariant(fun::Fun{D,P,R,C,S,T})::Bool where {D,P,R,C,S,T}
    D::Int
    P::PrimalDual
    R::Int
    0 ≤ R ≤ D || return false
    invariant(fun.manifold) || return false
    length(fun.values) == nsimplices(fun.manifold, R) || return false
    return true
end

# Comparison

function Base.:(==)(f::F, g::F) where {F<:Fun}
    @assert f.manifold == g.manifold
    f ≡ g && return true
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
function Base.rand(::Type{Fun{D,P,R,C,S,T}},
                   manifold::Manifold{D,C,S}) where {D,P,R,C,S,T}
    return Fun{D,P,R,C,S}(manifold,
                          IDVector{R}(rand(T, nsimplices(manifold, R))))
end

# Functions are a collection

Base.eltype(::Type{<:Fun{<:Any,<:Any,<:Any,<:Any,<:Any,T}}) where {T} = T
Base.eltype(f::Fun) = eltype(typeof(f))
Base.isempty(f::Fun) = isempty(f.values)
Base.iterate(f::Fun, state...) = iterate(f.values, state...)
Base.length(f::Fun) = length(f.values)

# function Base.map(op, f::Fun{D,P,R}, gs::Fun{D,P,R}...) where {D,P,R}
#     @assert all(f.manifold == g.manifold for g ∈ gs)
#     return Fun{D,P,R}(f.manifold,
#                         map(op, f.values, (g.values for g ∈ gs)...))
#     # r1 = map(op, f.values[1], (g.values[1] for g ∈ gs)...))
#     # T = typeof(r1)
#     # Fun{D, P, R, S, T}(f.manifold, map(op, f.values, (g.values for g ∈ gs)...))
# end
function Base.map(op, f::Fun{D,P,R,C,S},
                  gs::Fun{D,P,R,C,S}...) where {D,P,R,C,S}
    @assert all(g.manifold == f.manifold for g in gs)
    U = typeof(op(zero(eltype(f)), (zero(eltype(g)) for g in gs)...))
    return Fun{D,P,R,C,S,U}(f.manifold,
                            map(op, f.values, (g.values for g in gs)...))
end
function Base.reduce(op, f::Fun{D,P,R,C,S}, gs::Fun{D,P,R,C,S}...;
                     kw...) where {D,P,R,C,S}
    @assert all(g.manifold == f.manifold for g in gs)
    return reduce(op, f.values, (g.values for g in gs)...; kw...)
end

# Functions are an abstract vector

Base.IndexStyle(::Type{<:Fun}) = IndexStyle(Vector)
Base.axes(f::Fun) = axes(f.values)
Base.axes(f::Fun, dir) = axes(f.values, dir)
Base.eachindex(f::Fun) = eachindex(f.values)
Base.getindex(f::Fun, inds...) = getindex(f.values, inds...)
Base.ndims(::Fun) = 1
Base.size(f::Fun) = size(f.values)
Base.size(f::Fun, dims) = size(f.values, dims)
Base.stride(f::Fun, k) = stride(f.values, k)
Base.strides(f::Fun) = strides(f.values)

# Functions are a vector space

function Base.zero(::Type{Fun{D,P,R,C,S,T}},
                   manifold::Manifold{D,C,S}) where {D,P,R,C,S,T}
    nelts = nsimplices(manifold, R)
    return Fun{D,P,R}(manifold, IDVector{R}(zeros(T, nelts)))
end
function Base.zero(::Type{Fun{D,P,R,C,S}},
                   manifold::Manifold{D,C,S}) where {D,P,R,C,S,T}
    return zero(Fun{D,P,R,C,S,Bool}, manifold)
end
Base.zero(f::Fun) = zero(typeof(f), f.manifold)
Base.iszero(f::Fun) = iszero(f.values)

# LinearAlgebra.norm(f::Fun) = norm(f.values)
# LinearAlgebra.norm(f::Fun, p::Real) = norm(f.values, p)

export unit
function Forms.unit(::Type{Fun{D,P,R,C,S,T}}, manifold::Manifold{D,C,S},
                    n::Int) where {D,P,R,C,S,T}
    nelts = nsimplices(manifold, R)
    @assert 1 ≤ n ≤ nelts
    # return Fun{D,P,R,C,S}(manifold, sparsevec([n], [one(T)],nelts))
    return Fun{D,P,R,C,S}(manifold, IDVector{R}(T[i == n for i in 1:nelts]))
end

function Base.:+(f::Fun{D,P,R,C,S}) where {D,P,R,C,S}
    return Fun{D,P,R,C,S}(f.manifold, +f.values)
end

function Base.:-(f::Fun{D,P,R,C,S}) where {D,P,R,C,S}
    return Fun{D,P,R,C,S}(f.manifold, -f.values)
end

function Base.:+(f::Fun{D,P,R,C,S}, g::Fun{D,P,R,C,S}) where {D,P,R,C,S}
    @assert f.manifold == g.manifold
    return Fun{D,P,R,C,S}(f.manifold, f.values + g.values)
end

function Base.:-(f::Fun{D,P,R,C,S}, g::Fun{D,P,R,C,S}) where {D,P,R,C,S}
    @assert f.manifold == g.manifold
    return Fun{D,P,R,C,S}(f.manifold, f.values - g.values)
end

function Base.:*(a::Number, f::Fun{D,P,R,C,S}) where {D,P,R,C,S}
    return Fun{D,P,R,C,S}(f.manifold, a * f.values)
end

function Base.:\(a::Number, f::Fun{D,P,R,C,S}) where {D,P,R,C,S}
    return Fun{D,P,R,C,S}(f.manifold, a \ f.values)
end

function Base.:*(f::Fun{D,P,R,C,S}, a::Number) where {D,P,R,C,S}
    return Fun{D,P,R,C,S}(f.manifold, f.values * a)
end

function Base.:/(f::Fun{D,P,R,C,S}, a::Number) where {D,P,R,C,S}
    return Fun{D,P,R,C,S}(f.manifold, f.values / a)
end

Base.:~(f::Fun) = map(~, f)
Base.:&(f::Fun, g::Fun) = map(&, f, g)
Base.:|(f::Fun, g::Fun) = map(|, f, g)

# Functions are a category

export id
function id(::Type{<:Fun{D,P,0,C,S,SVector{C,S}}},
            manifold::Manifold{D,C,S}) where {D,P,C,S}
    return Fun{D,P,0,C,S}(manifold, get_coords(manifold))
end

function Base.conj(f::Fun{D,P,R,C,S}) where {D,P,R,C,S}
    return Fun{D,P,R,C,S}(f.manifold, conj(f.values))
end

# composition?

end
