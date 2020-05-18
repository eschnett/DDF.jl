module Defs

using ComputedFieldTypes
using Grassmann
using SparseArrays
using StaticArrays



export invariant
function invariant end

export unit
function unit end



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



# Allow constructing static arrays from generators
# <https://github.com/JuliaArrays/StaticArrays.jl/issues/791>

@generated function StaticArrays.SVector{N, T}(gen::Base.Generator) where {N, T}
    stmts = [:(Base.@_inline_meta)]
    args = []
    iter = :(iterate(gen))
    for i in 1:N
        el = Symbol(:el, i)
        push!(stmts, :(($el,st) = $iter))
        push!(args, el)
        iter = :(iterate(gen,st))
    end
    push!(stmts, :(SVector{N,T}($(args...))))
    Expr(:block, stmts...)
end

@generated function StaticArrays.SVector{N}(gen::Base.Generator) where {N}
    stmts = [:(Base.@_inline_meta)]
    args = []
    iter = :(iterate(gen))
    for i in 1:N
        el = Symbol(:el, i)
        push!(stmts, :(($el,st) = $iter))
        push!(args, el)
        iter = :(iterate(gen,st))
    end
    push!(stmts, :(SVector{N}($(args...))))
    Expr(:block, stmts...)
end

@generated function StaticArrays.SMatrix{M, N, T}(gen::Base.Generator) where {M, N, T}
    stmts = [:(Base.@_inline_meta)]
    args = []
    iter = :(iterate(gen))
    for j in 1:N, i in 1:M
        el = Symbol(:el, i, :x, j)
        push!(stmts, :(($el,st) = $iter))
        push!(args, el)
        iter = :(iterate(gen,st))
    end
    push!(stmts, :(SMatrix{M, N, T}($(args...))))
    Expr(:block, stmts...)
end

@generated function StaticArrays.SMatrix{M, N}(gen::Base.Generator) where {M, N}
    stmts = [:(Base.@_inline_meta)]
    args = []
    iter = :(iterate(gen))
    for j in 1:N, i in 1:M
        el = Symbol(:el, i, :x, j)
        push!(stmts, :(($el,st) = $iter))
        push!(args, el)
        iter = :(iterate(gen,st))
    end
    push!(stmts, :(SMatrix{M, N}($(args...))))
    Expr(:block, stmts...)
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

# Implement a unary version of ∧
# <https://github.com/chakravala/Grassmann.jl/issues/69>
Grassmann.:∧(x::Chain) = x
Grassmann.:∧(x::Simplex) = x
Grassmann.:∧(x::MultiVector) = x



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
    A = SMatrix{R+1,R+1}(
        i<=R && j<=R
        ? 2*scalar(xs[i]⋅xs[j]).v
        : i==j ? zero(T) : one(T)
        for i in 1:R+1, j in 1:R+1)
    b = SVector{R+1}(i<=R ? scalar(xs[i]⋅xs[i]).v : one(T) for i in 1:R+1)
    c = A \ b
    cc = sum(c[i]*xs[i] for i in 1:R)
    cc::Chain{V, 1, T}
end



export regular_simplex
# Generate the coordinate positions for a regular simplex with edge
# length 1.
#
# The algorithm proceeds recursively. A 0D simplex is a point. A
# D-simplex is a D-1-simplex that is shifted down along the new axis,
# plus a new point on the new axis.
function regular_simplex(::Val{D}, ::Type{T}) where {D, T}
    D::Integer
    @assert D >= 0
    T::Type
    S = Signature(D)
    V = SubManifold(S)
    B = Λ(S)
    N = D+1
    # D == 0 && return SVector{N}(Chain{V,1,T}())
    @assert D > 0
    D == 1 && return SVector{N}(Chain{V,1}(T(-1)/2), Chain{V,1}(T(1)/2))
    s0 = regular_simplex(Val(D-1), T)
    s0::SVector{N-1, fulltype(Chain{SubManifold(Signature(D-1)),1,T})}
    # Choose height so that edge length is 1
    z = sqrt(1 - scalar(abs2(s0[1])).v)
    z0 = -z/(D+1)
    extend(x) = Chain{V,1}(x.v..., z0)
    s = SVector{N}(extend.(s0)..., Chain((z+z0)*B.v(D)))
    s::SVector{N, fulltype(Chain{V,1,T})}
end

end
