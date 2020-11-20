module Algorithms

using DifferentialForms
using StaticArrays

export barycentre
barycentre(xs::SVector{N,<:Form{D,1,T}}) where {N,D,T} = sum(xs) / length(xs)
barycentre(xs::SVector{N,SVector{D,T}}) where {N,D,T} = sum(xs) / length(xs)

# function circumcentre1(xs::SVector{R, <:Chain{V, 1, T}}) where {R, V, T}
#     # G. Westendorp, A formula for the N-circumsphere of an N-simplex,
#     # <https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm>,
#     # April 2013.
#     @assert iseuclidean(V)
#     D = ndims(V)
#     @assert R == D + 1
# 
#     # Convert Euclidean to conformal basis
#     cxs = conformal.(xs)
#     # Circumsphere (this formula is why we are using conformal GA)
#     X = ∧(cxs)
#     # Hodge dual
#     sX = ⋆X
#     # Euclidean part is centre
#     cc = euclidean(sX)
# 
#     # Calculate radius
#     # TODO: Move this into a test case
#     r2 = scalar(abs2(cc)).v - 2 * sX.v[1]
#     # Check radii
#     for i ∈ 1:R
#         ri2 = scalar(abs2(xs[i] - cc)).v
#         @assert abs(ri2 - r2) ≤ T(1.0e-12) * r2
#     end
# 
#     cc::Chain{V, 1, T}
# end

export circumcentre
function circumcentre(xs::SVector{N,<:Form{D,1,T}}) where {N,D,T}
    D::Int
    @assert D ≥ 0
    N::Int
    @assert N ≥ 1
    @assert N ≤ D + 1
    # See arXiv:1103.3076v2 [cs.RA], section 10.1
    A = SMatrix{N + 1,N + 1}(i ≤ N && j ≤ N ? 2 * (xs[i] ⋅ xs[j])[] :
                             i == j ? zero(T) : one(T)
                             for i in 1:(N + 1), j in 1:(N + 1))
    b = SVector{N + 1}(i ≤ N ? (xs[i] ⋅ xs[i])[] : one(T) for i in 1:(N + 1))
    c = A \ b
    cc = sum(c[i] * xs[i] for i in 1:N)
    return cc::Form{D,1,T}
end

function circumcentre(xs::SVector{N,<:Form{D,1,T}},
                      ws::SVector{N,<:Form{D,0,T′}}) where {N,D,T,T′}
    # See arXiv:math/0508188, section 3.2:
    # See arXiv:1103.3076v2 [cs.RA], section 10.1:
    #
    # C = Ci xi + Cj xj + Ck xk
    # (C - xi) (C - xi) = r^2 + wi
    # (C - xj) (C - xj) = r^2 + wj
    # (C - xk) (C - xk) = r^2 + wk
    # Ci + Cj + Ck = 1
    # 
    # (Ci xi + Cj xj + Ck xk - xi) (Ci xi + Cj xj + Ck xk - xi) = r^2 + wi
    # (Ci xi + Cj xj + Ck xk - xj) (Ci xi + Cj xj + Ck xk - xj) = r^2 + wj
    # (Ci xi + Cj xj + Ck xk - xk) (Ci xi + Cj xj + Ck xk - xk) = r^2 + wk
    # Ci + Cj + Ck = 1
    # 
    # C^2 - 2 xi (Ci xi + Cj xj + Ck xk) + xi^2 = r^2 + wi
    # C^2 - 2 xj (Ci xi + Cj xj + Ck xk) + xj^2 = r^2 + wj
    # C^2 - 2 xk (Ci xi + Cj xj + Ck xk) + xk^2 = r^2 + wk
    # Ci + Cj + Ck = 1
    # 
    # 2 xi (Ci xi + Cj xj + Ck xk) + (r^2 - C^2) = xi^2 - wi
    # 2 xj (Ci xi + Cj xj + Ck xk) + (r^2 - C^2) = xj^2 - wj
    # 2 xk (Ci xi + Cj xj + Ck xk) + (r^2 - C^2) = xk^2 - wk
    # Ci + Cj + Ck = 1

    D::Int
    @assert D ≥ 0
    N::Int
    @assert N ≥ 1
    @assert N ≤ D + 1
    # See arXiv:1103.3076v2 [cs.RA], section 10.1
    A = SMatrix{N + 1,N + 1}(i ≤ N && j ≤ N ? 2 * (xs[i] ⋅ xs[j])[] :
                             i == j ? zero(T) : one(T)
                             for i in 1:(N + 1), j in 1:(N + 1))
    b = SVector{N + 1}(i ≤ N ? (xs[i] ⋅ xs[i] - ws[i])[] : one(T)
                       for i in 1:(N + 1))
    c = A \ b
    cc = sum(c[i] * xs[i] for i in 1:N)
    return cc::Form{D,1}
end

export volume
"""
Unsigned volume
"""
function volume(xs::SVector{N,<:Form{D,1,T}}) where {N,D,T}
    D::Int
    @assert D ≥ 0
    N::Int
    @assert N ≥ 1
    @assert N ≤ D + 1
    ys = map(x -> x - xs[1], deleteat(xs, 1))
    vol0 = norm(∧(ys))
    if T <: Rational
        vol = rationalize(typeof(zero(T).den), vol0; tol=sqrt(eps(vol0)))
    else
        vol = vol0
    end
    vol::T
    vol /= factorial(N - 1)
    return vol::T
end
function volume(xs::SVector{N,SVector{D,T}}) where {N,D,T}
    return volume(SVector{N}(Form{D,1,T}(x) for x in xs))
end

export signed_volume
"""
Signed volume
"""
function signed_volume(xs::SVector{N,<:Form{D,1,T}},
                       signature::Int=1) where {N,D,T}
    D::Int
    @assert D ≥ 0
    N::Int
    R = N - 1
    @assert 0 ≤ R ≤ D
    # TODO: add metric signatures to DifferentialForms.jl
    # TODO: add outermorphisms to DifferentialForms.jl
    # TODO: move Plotting into its own module; just provide functions
    #       to obtain vertices, connectivity, etc.
    ys = map(x -> x - xs[1], deleteat(xs, 1))
    η = SVector{D,T}(a == D ? signature : 1 for a in 1:D)
    ys′ = map(y -> Form{D,1,T}(η .* y.elts), ys)
    y = ∧(ys)
    y′ = ∧(ys′)
    v2 = (y′ ⋅ y)[]::T
    sv2 = sign(v2)::T
    v = (sv2 * sqrt(abs(v2)))::T
    return v / factorial(R)
end

################################################################################

# See also <https://github.com/JaneliaSciComp/Morton.jl>
export morton
function morton(xs::SVector{D,T}, xmin::SVector{D,T},
                xmax::SVector{D,T}) where {U<:Unsigned,D,T}
    return morton(UInt64, xs, xmin, xmax)
end
function morton(::Type{U}, xs::SVector{D,T}, xmin::SVector{D,T},
                xmax::SVector{D,T}) where {U<:Unsigned,D,T}
    D == 0 && return morton(SVector{D,U}())
    y = (xs - xmin) ./ (xmax - xmin)
    nbits = min(sizeof(U) - 1, 8 * sizeof(U) ÷ D)
    nbins = U(1) << nbits
    i = clamp.(floor.(U, nbins .* y), 0, nbins - 1)
    return morton(i)
end
function morton(i::SVector{D,U}) where {D,U<:Unsigned}
    D == 0 && return U(0)
    return reduce(|, SVector{D,U}(spread(Val(D), i[d]) << (d - 1) for d in 1:D))
end

spread(::Val{1}, i::Unsigned) = i
function spread(::Val{2}, i::UInt64)
    # can use 64 ÷ 2 = 32 bits
    @assert i ==
            i &
            0b0000000000000000000000000000000011111111111111111111111111111111
    i = i & 0b0000000000000000000000000000000011111111111111111111111111111111
    i = ((i << 16) | i) &
        0b0000000000000000111111111111111100000000000000001111111111111111
    i = ((i << 08) | i) &
        0b0000000011111111000000001111111100000000111111110000000011111111
    i = ((i << 04) | i) &
        0b0000111100001111000011110000111100001111000011110000111100001111
    i = ((i << 02) | i) &
        0b0011001100110011001100110011001100110011001100110011001100110011
    i = ((i << 01) | i) &
        0b0101010101010101010101010101010101010101010101010101010101010101
    return i
end
function spread(::Val{3}, i::UInt64)
    # can use 64 ÷ 3 = 21 bits
    @assert i ==
            i &
            0b000000000000000000000000000000000000000000111111111111111111111
    i = i & 0b000000000000000000000000000000000000000000111111111111111111111
    i = ((i << 32) | i) &
        0b000000000011111000000000000000000000000000000001111111111111111
    i = ((i << 16) | i) &
        0b000000000011111000000000000000011111111000000000000000011111111
    i = ((i << 08) | i) &
        0b001000000001111000000001111000000001111000000001111000000001111
    i = ((i << 04) | i) &
        0b001000011000011000011000011000011000011000011000011000011000011
    i = ((i << 02) | i) &
        0b001001001001001001001001001001001001001001001001001001001001001
    return i
end
function spread(::Val{4}, i::UInt64)
    # can use 64 ÷ 4 = 16 bits
    @assert i ==
            i &
            0b0000000000000000000000000000000000000000000000001111111111111111
    i = i & 0b0000000000000000000000000000000000000000000000001111111111111111
    i = ((i << 24) | i) &
        0b0000000000000000000000001111111100000000000000000000000011111111
    i = ((i << 12) | i) &
        0b0000000000001111000000000000111100000000000011110000000000001111
    i = ((i << 06) | i) &
        0b0000001100000011000000110000001100000011000000110000001100000011
    i = ((i << 03) | i) &
        0b0001000100010001000100010001000100010001000100010001000100010001
    return i
end
function spread(::Val{5}, i::UInt64)
    # can use 64 ÷ 5 = 12 bits
    @assert i ==
            i & 0b000000000000000000000000000000000000000000000000111111111111
    i = i & 0b000000000000000000000000000000000000000000000000111111111111
    i = ((i << 32) | i) &
        0b000000000000000011110000000000000000000000000000000011111111
    i = ((i << 16) | i) &
        0b000000000000000011110000000000000000111100000000000000001111
    i = ((i << 08) | i) &
        0b000000001100000000110000000011000000001100000000110000000011
    i = ((i << 04) | i) &
        0b000010000100001000010000100001000010000100001000010000100001
    return i
end
function spread(::Val{6}, i::UInt64)
    # can use 64 ÷ 6 = 10 bits
    @assert i ==
            i & 0b000000000000000000000000000000000000000000000000001111111111
    i = i & 0b000000000000000000000000000000000000000000000000001111111111
    i = ((i << 40) | i) &
        0b000000000011000000000000000000000000000000000000000011111111
    i = ((i << 20) | i) &
        0b000000000011000000000000000000001111000000000000000000001111
    i = ((i << 10) | i) &
        0b000000000011000000000011000000000011000000000011000000000011
    i = ((i << 05) | i) &
        0b000001000001000001000001000001000001000001000001000001000001
    return i
end

end
