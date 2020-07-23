module Forms

using ComputedFieldTypes
using Grassmann
using LinearAlgebra
using StaticArrays

using ..Defs

function VecType(D::Int, R::Int, T)
    @assert D >= 0
    @assert 0 <= R <= D
    return Chain{SubManifold(Signature(D)),R,T}
end

export Form
@computed struct Form{D,R,T}
    vec::fulltype(VecType(D, R, T))

    Form{D,R,T}(xs::Chain{V,R,T}) where {D,V,R,T} = new{D,R,T}(xs)
    Form{D,R,T}(xs::Chain{V,R}) where {D,V,R,T} = Form{D,R,T}(Chain{T}(xs))
    Form{D,R}(xs::Chain{V,R,T}) where {D,V,R,T} = Form{D,R,T}(xs)
    Form{D}(xs::Chain{V,R}) where {D,V,R} = Form{D,R}(xs)
    Form(xs::Chain{V,R}) where {V,R} = Form{ndims(V),R}(xs)
end

# Construct Forms from their components
function Form{D,R}(xs::SVector) where {D,R}
    V = SubManifold(Signature(D))
    return Form{D,R}(Chain{V,R}(xs))
end
function Form{D,R,T}(xs::SVector) where {D,R,T}
    V = SubManifold(Signature(D))
    return Form{D,R,T}(Chain{V,R}(xs))
end
Form{D,R}(xs::Tuple) where {D,R} = Form{D,R}(SVector(xs))
Form{D,R,T}(xs::Tuple) where {D,R,T} = Form{D,R,T}(SVector(xs))

# TODO: Add conversions to Chain etc.
# Base.convert(::Type{SA}, x::Form) where {SA <: StaticArray} = x.vec.v::SA

# # Construct scalars from scalars
# export Scalar
# const Scalar{D, T} = Form{D, 0, T}
# Scalar{D, T}(x) where {D, T} = Form{D, 0}(SVector{1, T}(x))
# Scalar{D}(x) where D = Form{D, 0}(SVector(x))

# # Construct vectors from multiple scalars
# export OneForm
# const OneForm{D, T} = Form{D, 1, T}
# OneForm{D, T}(x) where {D, T} = Form{D, 0}(SVector{D, T}(x))
# OneForm{D}(x) where D = Form{D, 0}(SVector(x))

Base.convert(::Type{<:T}, x::Form{D,0,T}) where {D,T} = x[]
Base.convert(::Type{SVector}, x::Form{D,1,T}) where {D,T} = x.vec.v
Base.convert(::Type{SVector{D}}, x::Form{D,1,T}) where {D,T} = x.vec.v
Base.convert(::Type{SVector{D,T}}, x::Form{D,1,T}) where {D,T} = x.vec.v

export fdim
fdim(::Type{<:Form{D}}) where {D} = D
fdim(::F) where {F<:Form} = fdim(F)
export frank
frank(::Type{<:Form{D,R}} where {D}) where {R} = R
frank(::F) where {F<:Form} = frank(F)

# I/O

function Base.show(io::IO, x::Form)
    print(io, "{$(fdim(x)),$(frank(x))}[")
    for a = 1:length(x)
        a != 1 && print(io, ",")
        print(io, x.vec[a])
    end
    return print(io, "]")
end

# Type promotion

# TODO: use type promotion instead of generic definitions?

# Collection

function Base.IteratorSize(::Type{<:Form{D,R,T}}) where {D,R,T}
    return Base.IteratorSize(VecType(D, R, T))
end
function Base.IteratorEltype(::Type{<:Form{D,R,T}}) where {D,R,T}
    return Base.IteratorEltype(VecType(D, R, T))
end
Base.eltype(x::Form) = eltype(x.vec)
Base.isempty(x::Form) = isempty(x.vec)
Base.iterate(x::Form, state...) = iterate(x.vec, state...)
Base.length(x::Form) = length(x.vec)
Base.ndims(x::Form) = ndims(x.vec)
Base.size(x::Form) = size(x.vec)
function Base.map(f, x::Form{D,R}, ys::Form{D,R}...) where {D,R}
    return Form{D,R}(map(f, x.vec, map(y -> y.vec, ys)...))
end

# Element access
function _getindex(x::Form{D,R}, is::SVector{R,Int}) where {D,R}
    js, s = sort_perm(is)
    for r = 2:R
        js[r] == js[r-1] && return zero(T)
    end

    B = Λ(Signature(D))
    r = (x.vec⋅(isempty(is) ? B.v : B.v(is...))).v[1]
    return isodd(s) ? -r : r
end
@inline function _getindex(x::Form{D,R}, is::SVector{R}) where {D,R}
    return _getindex(x, SVector{R,Int}(is))
end
@inline Base.getindex(x::Form, is...) = _getindex(x, SVector(is...))
@inline Base.getindex(x::Form) = _getindex(x, SVector{0,Int}())

# Comparison

Base.:(==)(x::Form, y::Form) = x.vec == y.vec
Base.isless(x::Form, y::Form) = isless(x.vec, y.vec)

# Field

Base.one(::Type{<:Form{D,0,T}}) where {D,T} = Form{D,0}((one(T),))
Base.one(::F) where {F<:Form} = one(F)

# Nullary functions
for fun in [:rand, :zero]
    @eval begin
        function Base.$fun(::Type{<:Form{D,R,T}}) where {D,R,T}
            return Form($fun(VecType(D, R, T)))
        end
        Base.$fun(::F) where {F<:Form} = $fun(F)
    end
end

# Unary functions
for fun in [:+, :-, :~, :conj, :inv]
    @eval Base.$fun(x::Form) = Form($fun(x.vec))
end
for fun in [:⋆]
    @eval begin
        export $fun
        Grassmann.$fun(x::Form) = Form($fun(x.vec))
    end
end
export invhodge
invhodge(x::Form{D,R}) where {D,R} = bitsign(R * (D - R)) * ⋆x
Base.inv(::typeof(⋆)) = invhodge
for fun in [:transpose]
    @eval begin
        export $fun
        LinearAlgebra.$fun(x::Form) = Form($fun(x.vec))
    end
end
Base.:^(x::Form, n::Integer) = Form(^(x.vec, n))
Base.abs(x::Form) = abs(x.vec).v[1]
Base.abs2(x::Form) = abs2(x.vec).v[1]
Base.iszero(x::Form) = iszero(x.vec)
LinearAlgebra.norm(x::Form; p) = norm(x.vec; p = p)

# Binary functions
for fun in [:+, :-]             # :*, :/, :\
    @eval function Base.$fun(x::Form{D,R}, y::Form{D,R}) where {D,R}
        return Form($fun(x.vec, y.vec))
    end
end
for fun in [:∧, :∨, :⋅, :×]
    @eval begin
        export $fun
        function Grassmann.$fun(x::Form{D}, ys::Form{D}...) where {D}
            return Form(Chain($fun(x.vec, map(y -> y.vec, ys)...)))
        end
        Grassmann.$fun(x::Form) = x
    end
end

# Vector space

# Binary functions
Base.:*(a::Number, x::Form) = Form(*(a, x.vec))
Base.:*(x::Form, a::Number) = Form(*(x.vec, a))
Base.:\(a::Number, x::Form) = Form(\(a, x.vec))
Base.:/(x::Form, a::Number) = Form(/(x.vec, a))

Base.:*(a::T, x::Form{D,R,T}) where {D,R,T} = Form(*(a, x.vec))
Base.:*(x::Form{D,R,T}, a::T) where {D,R,T} = Form(*(x.vec, a))
Base.:\(a::T, x::Form{D,R,T}) where {D,R,T} = Form(\(a, x.vec))
Base.:/(x::Form{D,R,T}, a::T) where {D,R,T} = Form(/(x.vec, a))

# Sum space
export ⊗
function ⊗(x::Form{D1,1}, y::Form{D2,1}) where {D1,D2}
    D1::Int
    @assert D1 >= 0
    D2::Int
    @assert D2 >= 0
    return Form{D1 + D2,1}(SVector(x.vec.v..., y.vec.v...))
end
function ⊗(x::Form{D1,1,T}, a::T) where {D1,T}
    D1::Int
    @assert D1 >= 0
    return Form{D1 + 1,1}(SVector(x.vec.v..., a))
end
function ⊗(a::T, x::Form{D1,1,T}) where {D1,T}
    D1::Int
    @assert D1 >= 0
    return Form{D1 + 1,1}(SVector(a, x.vec.v...))
end

# Subspace
function Base.getindex(x::Form{D1,1}, i::SVector{D2,Int}) where {D1,D2}
    D1::Int
    @assert D1 >= 0
    D2::Int
    @assert D1 - D2 >= 0
    return Form{D1 - D2,1}(x.vec.v[i])
end
@inline function Base.getindex(x::Form{D1,1}, i::SVector{D2}) where {D1,D2}
    return getindex(x, SVector{D2,Int}(i))
end
@inline getindex(x::Form, i::Tuple) = getindex(x, SVector(i...))
@inline getindex(x::Form, i::Tuple{}) = getindex(x, SVector{0,Int}())

# # Homogeneous and conformal geometries
# 
# export iseuclidean
# iseuclidean(::Chain{V}) where {V} = iseuclidean(V)
# iseuclidean(V::SubManifold) = !hasinf(V) && !hasorigin(V)
# 
# export ishomogeneous
# ishomogeneous(::Chain{V}) where {V} = ishomogeneous(V)
# ishomogeneous(V::SubManifold) = hasinf(V) && !hasorigin(V)
# 
# export isconformal
# isconformal(::Chain{V}) where {V} = isconformal(V)
# isconformal(V::SubManifold) = hasconformal(V)
# 
# @generated function prolong(::Val{W}, x::Chain{V,1,T}) where {W, V, T}
#     W::SubManifold
#     V::SubManifold
#     nins = ndims(W) - ndims(V)
#     @assert nins >= 0
#     quote
#         Chain{W,1}($([d<=nins ? :(zero(T)) : :(x[$(d-nins)])
#                       for d in 1:ndims(W)]...))::Chain{W,1,T}
#     end
# end
# 
# @generated function restrict(::Val{W}, x::Chain{V,1,T}) where {W, V, T}
#     W::SubManifold
#     V::SubManifold
#     nskip = ndims(V) - ndims(W)
#     @assert nskip >= 0
#     quote
#         Chain{W,1}($([:(x[$(d+nskip)]) for d in 1:ndims(W)]...))
#     end
# end
# 
# export euclidean
# @generated function euclidean(x::Chain{V,1,T}) where {V, T}
#     @assert iseuclidean(V) || ishomogeneous(V) || isconformal(V)
#     iseuclidean(V) && return :x
#     nskip = ishomogeneous(V) + 2*isconformal(V)
#     W = SubManifold(Signature(ndims(V)-nskip))
#     U = typeof(inv(one(T)))
#     quote
#         y = ↓(x)
#         @assert isvector(y)
#         z = Chain(vector(y))::Chain{V,1,$U}
#         restrict($(Val(W)), z)::Chain{$W,1,$U}
#     end
# end
# 
# export homogeneous
# @generated function homogeneous(x::Chain{V,1,T}) where {V, T}
#     @assert iseuclidean(V) || ishomogeneous(V) || isconformal(V)
#     ishomogeneous(V) && return :x
#     U = typeof(inv(one(T)))
#     if iseuclidean(V)
#         W = SubManifold(Signature(ndims(V)+1, 1))
#         quote
#             y = ↑(prolong($(Val(W)), x))
#             @assert isvector(y)
#             Chain(vector(y))::Chain{$W,1,$U}
#         end
#     else
#         W = SubManifold(Signature(ndims(V)-1, 1))
#         quote
#             y = ↓(x)
#             @assert isvector(y)
#             z = Chain(vector(y))::Chain{V,1,$U}
#             restrict($(Val(W)), z)::Chain{$W,1,$U}
#         end
#     end
# end
# 
# export conformal
# @generated function conformal(x::Chain{V,1,T}) where {V, T}
#     @assert iseuclidean(V) || ishomogeneous(V) || isconformal(V)
#     isconformal(V) && return :x
#     nins = 2*iseuclidean(V) + ishomogeneous(V)
#     W = SubManifold(Signature(ndims(V)+nins, 1, 1))
#     U = typeof(inv(one(T)))
#     quote
#         y = ↑(prolong($(Val(W)), x))
#         @assert isvector(y)
#         Chain(vector(y))::Chain{$W,1,$U}
#     end
# end

end
