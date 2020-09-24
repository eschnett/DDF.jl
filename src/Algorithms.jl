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
#     for i in 1:R
#         ri2 = scalar(abs2(xs[i] - cc)).v
#         @assert abs(ri2 - r2) <= T(1.0e-12) * r2
#     end
# 
#     cc::Chain{V, 1, T}
# end

export circumcentre
function circumcentre(xs::SVector{N,<:Form{D,1,T}}) where {N,D,T}
    D::Int
    @assert D >= 0
    N::Int
    @assert N >= 1
    @assert N <= D + 1
    # See arXiv:1103.3076v2 [cs.RA], section 10.1
    A = SMatrix{N + 1,N + 1}(i <= N && j <= N ? 2 * (xs[i] ⋅ xs[j])[] :
                             i == j ? zero(T) : one(T)
                             for i in 1:(N + 1), j in 1:(N + 1))
    b = SVector{N + 1}(i <= N ? (xs[i] ⋅ xs[i])[] : one(T) for i in 1:(N + 1))
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
    @assert D >= 0
    N::Int
    @assert N >= 1
    @assert N <= D + 1
    # See arXiv:1103.3076v2 [cs.RA], section 10.1
    A = SMatrix{N + 1,N + 1}(i <= N && j <= N ? 2 * (xs[i] ⋅ xs[j])[] :
                             i == j ? zero(T) : one(T)
                             for i in 1:(N + 1), j in 1:(N + 1))
    b = SVector{N + 1}(i <= N ? (xs[i] ⋅ xs[i] - ws[i])[] : one(T)
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
    @assert D >= 0
    N::Int
    @assert N >= 1
    @assert N <= D + 1
    ys = map(x -> x - xs[1], deleteat(xs, 1))
    vol0 = norm(∧(ys))
    if T <: Rational
        vol = rationalize(typeof(zero(T).den), vol0; tol = sqrt(eps(vol0)))
    else
        vol = vol0
    end
    vol::T
    vol /= factorial(N - 1)
    return vol::T
end

export signed_volume
"""
Signed volume
"""
function signed_volume(xs::SVector{N,<:Form{D,1,T}}) where {N,D,T}
    D::Int
    @assert D >= 0
    N::Int
    R = N - 1
    @assert 0 <= R <= D
    ys = map(x -> x - xs[1], deleteat(xs, 1))
    if isempty(ys)
        vol = one(Form{D,0,T})  # 0
    else
        y = ∧(ys)               # R
        ny = norm(y)
        ny = ifelse(ny == 0, one(ny), ny)
        n = y / ny              # R
        vol = n ⋅ y             # 0
    end
    return vol[] / factorial(R)
end

# export dualvolume
# function dualvolume()
#     # Calculate circumcentric dual volumes
#     # [1198555.1198667, page 5]
#     dualvolumes = Dict{Int,Fun{D,Dl,R,T} where {R}}()
#     for R in D:-1:0
#         if R == D
#             values = ones(T, size(R, topo))
#         else
#             bnds = topo.boundaries[R + 1]
#             values = zeros(T, size(R, topo))
#             sis = topo.simplices[R]::Vector{Simplex{R + 1,Int}}
#             sjs = topo.simplices[R + 1]::Vector{Simplex{R + 2,Int}}
#             for (i, si) in enumerate(sis)
#                 # TODO: This is expensive
#                 js = findnz(bnds[i, :])[1]
#                 for j in js
#                     sj = sjs[j]
#                     b = dualvolumes[R + 1][j]
#                     # TODO: Calculate lower-rank circumcentres as
#                     # intersection between boundary and the line
#                     # connecting two simplices?
#                     # TODO: Cache circumcentres ahead of time
#                     @assert length(si.vertices) == R + 1
#                     @assert length(sj.vertices) == R + 2
#                     xsi = coords[si.vertices]
#                     cci = circumcentre(xsi)
#                     xsj = coords[sj.vertices]
#                     ccj = circumcentre(xsj)
#                     # TODO: Handle case where the volume should be
#                     # negative (i.e. when the volume circumcentre ccj
#                     # is on the "other" side of the face circumcentre
#                     # cci) (Is the previous statement correct?)
#                     h = abs(cci - ccj)
#                     values[i] += b * h / factorial(D - R)
#                 end
#             end
#         end
#         # @assert all(>(0), values)
#         vols = Fun{D,Dl,R,T}(topo, values)
#         dualvolumes[R] = vols
#     end
# end

end
