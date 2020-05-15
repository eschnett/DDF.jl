module Defs

using ComputedFieldTypes
using Grassmann
using SparseArrays
using StaticArrays



export invariant
function invariant end



export bitsign
bitsign(b::Bool) = b ? -1 : 1
bitsign(i::Integer) = bitsign(isodd(i))



export sort_perm
"""
    sort_perm

Sort and count permutations.
"""
function sort_perm(xs::SVector{D, T}) where {D, T}
    rs = xs
    s = false
    for i in 1:D
        imin = i
        # selection sort
        for j in i+1:D
            xs[j] < xs[imin] && (imin = j)
        end
        rs = setindex(rs, xs[imin], i)
        xs = setindex(xs, xs[i], imin)
        s = xor(s, isodd(i - imin))
    end
    # @assert issorted(rs)
    rs, s
end



export show_sparse
function show_sparse(io::IO, A::SparseMatrixCSC{T,I}) where {T, I}
    # Convert to CSR -- this is expensive!
    A = sparse(A')
    println(io,
            "$(A.n)×$(A.m) SparseMatrixCSC{$I,$T} ",
            "with $(length(A.nzval)) stored entries:")
    for i in 1:A.n
        jmin = A.colptr[i]
        jmax = A.colptr[i+1] - 1
        if jmin <= jmax
            print(io, "  [$i]:")
            for j in jmin:jmax
                print(io, " [$(A.rowval[j])]=$(A.nzval[j])")
            end
            println(io)
        end
    end
end
show_sparse(A::SparseMatrixCSC) = show_sparse(stdout, A)



export sarray
@generated function sarray(::Type{T}, f::F, ::Val{R}) where {T, F, R}
    R::Integer
    quote
        SArray{Tuple{$R}, T}($([:(f($i)::T) for i in 1:R]...))
    end
end
@generated function sarray(::Type{T}, f::F, ::Val{R1}, ::Val{R2}
                           ) where {T, F, R1, R2}
    R1::Integer
    R2::Integer
    quote
        SArray{Tuple{$R1, $R2}, T}(
            $([:(f($i, $j)::T) for i in 1:R1, j in 1:R2]...))
    end
end

@generated function Base.sum(::Type{T}, f::F, ::Val{R}) where {T, F, R}
    R::Integer
    if R <= 0
        quote
            zero(T)
        end
    elseif R == 1
        # Work around <https://github.com/chakravala/Grassmann.jl/issues/68>
        quote
            f(1)::T
        end
    else
        quote
            (+($([:(f($i)::T) for i in 1:R]...)))::T
        end
    end
end
@generated function Base.sum(::Type{T}, f::F, ::Val{R1}, ::Val{R2}
                             ) where {T, F, R1, R2}
    R1::Integer
    R2::Integer
    if R1 <= 0 || R2 <= 0
        quote
            zero(T)
        end
    elseif R1 == R2 == 1
        # Work around <https://github.com/chakravala/Grassmann.jl/issues/68>
        quote
            f(1, 1)::T
        end
    else
        quote
            (+($([:(f($i, $j)::T) for i in 1:R1, j in 1:R2]...)))::T
        end
    end
end



# Ensure type stability in some methods
# <https://github.com/chakravala/Grassmann.jl/issues/65>
function Grassmann.:↑(ω::T) where T<:TensorAlgebra
    V = Manifold(ω)
    !(hasinf(V)||hasorigin(V)) && (return ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        ((G.v∞*(one(valuetype(ω))/2))*ω^2+G.v∅)+ω
    else
        ω2 = ω^2
        iω2 = inv(ω2+1)
        (hasinf(V) ? G.v∞ : G.v∅)*(ω2-1)*iω2 + 2*iω2*ω
    end
end
function Grassmann.:↓(ω::T) where T<:TensorAlgebra
    V = Manifold(ω)
    !(hasinf(V)||hasorigin(V)) && (return ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        inv(one(valuetype(ω))*G.v∞∅)*(G.v∞∅∧ω)/(-ω⋅G.v∞)
    else
        b = hasinf(V) ? G.v∞ : G.v∅
        ((ω∧b)*b)/(1-b⋅ω)
    end
end

# Override (aka delete) harmful methods from Grassmann
# <https://github.com/chakravala/Grassmann.jl/issues/67>
Base.ndims(::Vector{Chain{V,G,T,X}} where {G,T,X}) where V = 1
# Base.parent(::Vector{Chain{V,G,T,X}} where {G,T,X}) where V = ???



export iseuclidean
@generated iseuclidean(::Chain{V}) where {V} = iseuclidean(V)
function iseuclidean(V::SubManifold)
    S = Signature(V)
    !hasinf(S) && !hasorigin(S)
end

export isprojective
@generated isprojective(::Chain{V}) where {V} = isprojective(V)
function isprojective(V::SubManifold)
    S = Signature(V)
    hasinf(S) && !hasorigin(S)
end

export isconformal
@generated isconformal(::Chain{V}) where {V} = isconformal(V)
function isconformal(V::SubManifold)
    S = Signature(V)
    hasinf(S) && hasorigin(S)
end

@generated function prolong(::Val{W}, x::Chain{V,1,T}) where {W, V, T}
    W::SubManifold
    V::SubManifold
    nins = ndims(W) - ndims(V)
    @assert nins >= 0
    quote
        Chain{W,1}($([d<=nins ? :(zero(T)) : :(x[$(d-nins)])
                      for d in 1:ndims(W)]...))::Chain{W,1,T}
    end
end

@generated function restrict(::Val{W}, x::Chain{V,1,T}) where {W, V, T}
    W::SubManifold
    V::SubManifold
    nskip = ndims(V) - ndims(W)
    @assert nskip >= 0
    quote
        Chain{W,1}($([:(x[$(d+nskip)]) for d in 1:ndims(W)]...))
    end
end

export euclidean
@generated function euclidean(x::Chain{V,1,T}) where {V, T}
    @assert iseuclidean(V) || isprojective(V) || isconformal(V)
    iseuclidean(V) && return :x
    nskip = isprojective(V) + 2*isconformal(V)
    W = SubManifold(Signature(ndims(V)-nskip))
    U = typeof(inv(one(T)))
    quote
        y = ↓(x)
        @assert isvector(y)
        z = Chain(vector(y))::Chain{V,1,$U}
        restrict($(Val(W)), z)::Chain{$W,1,$U}
    end
end

export projective
@generated function projective(x::Chain{V,1,T}) where {V, T}
    @assert iseuclidean(V) || isprojective(V) || isconformal(V)
    isprojective(V) && return :x
    U = typeof(inv(one(T)))
    if iseuclidean(V)
        W = SubManifold(Signature(ndims(V)+1, 1))
        quote
            y = ↑(prolong($(Val(W)), x))
            @assert isvector(y)
            Chain(vector(y))::Chain{$W,1,$U}
        end
    else
        W = SubManifold(Signature(ndims(V)-1, 1))
        quote
            y = ↓(x)
            @assert isvector(y)
            z = Chain(vector(y))::Chain{V,1,$U}
            restrict($(Val(W)), z)::Chain{$W,1,$U}
        end
    end
end

export conformal
@generated function conformal(x::Chain{V,1,T}) where {V, T}
    @assert iseuclidean(V) || isprojective(V) || isconformal(V)
    isconformal(V) && return :x
    nins = 2*iseuclidean(V) + isprojective(V)
    W = SubManifold(Signature(ndims(V)+nins, 1, 1))
    U = typeof(inv(one(T)))
    quote
        y = ↑(prolong($(Val(W)), x))
        @assert isvector(y)
        Chain(vector(y))::Chain{$W,1,$U}
    end
end



function circumcentre1(xs::SVector{R, <:Chain{V, 1, T}}) where {R, V, T}
    # G. Westendorp, A formula for the N-circumsphere of an N-simplex,
    # <https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm>,
    # April 2013.
    @assert iseuclidean(V)
    D = ndims(V)
    @assert R == D + 1

    # Convert Euclidean to conformal basis
    cxs = map(conformal, xs)
    # Circumsphere (this formula is why we are using conformal GA)
    X = ∧(cxs)
    # Hodge dual
    sX = ⋆X
    # Euclidean part is centre
    cc = euclidean(sX)

    # Calculate radius
    # TODO: Move this into a test case
    r2 = scalar(abs2(cc)).v - 2 * sX.v[1]
    # Check radii
    for i in 1:R
        ri2 = scalar(abs2(xs[i] - cc)).v
        @assert abs(ri2 - r2) <= T(1.0e-12) * r2
    end

    cc::Chain{V, 1, T}
end

export circumcentre
function circumcentre(xs::SVector{R, <:Chain{V, 1, T}}) where {R, V, T}
    # See arXiv:1103.3076v2 [cs.RA], section 10.1
    A = sarray(T,
               (i,j) -> i<=R && j<=R
               ? 2*scalar(xs[i]⋅xs[j]).v
               : i==j ? zero(T) : one(T),
               Val(R+1), Val(R+1))
    b = sarray(T, i -> i<=R ? scalar(xs[i]⋅xs[i]).v : one(T), Val(R+1))
    c = A \ b
    cc = sum(fulltype(Chain{V,1,T}), i -> c[i]*xs[i], Val(R))
    cc::Chain{V, 1, T}
end

end