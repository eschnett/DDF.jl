module Algorithms

using ComputedFieldTypes
using SparseArrays
using StaticArrays

using ..Forms



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
    # See arXiv:1103.3076v2 [cs.RA], section 10.1
    A = SMatrix{N + 1,N + 1}(
        i <= N && j <= N ? 2 * (xs[i]⋅xs[j])[] : i == j ? zero(T) : one(T)
        for i = 1:N+1, j = 1:N+1
    )
    b = SVector{N + 1}(i <= N ? (xs[i]⋅xs[i])[] : one(T) for i = 1:N+1)
    c = A \ b
    cc = sum(c[i] * xs[i] for i = 1:N)
    cc::Form{D,1,T}
end



export regular_simplex
# Generate the coordinate positions for a regular simplex with edge
# length 1.
#
# The algorithm proceeds recursively. A 0D simplex is a point. A
# D-simplex is a D-1-simplex that is shifted down along the new axis,
# plus a new point on the new axis.
function regular_simplex1(D::Int, ::Type{T}) where {T}
    @assert D >= 0
    N = D + 1
    # D == 0 && return SVector{N}(Chain{V,1,T}())
    @assert D > 0
    D == 1 && return SVector(Form{D,1}((T(-1) / 2,)), Form{D,1}((T(1) / 2,)))
    s0 = regular_simplex1(D - 1, T)
    # Choose height so that edge length is 1
    z = sqrt(1 - abs2(s0[1]))
    z0 = -z / (D + 1)
    s = SVector{N}(map(x -> x ⊗ z0, s0)..., zero(Form{D - 1,1,T}) ⊗ (z + z0))
    s::SVector{N,fulltype(Form{D,1,T})}
end
@generated function regular_simplex(::Type{<:Form{D,R,T}}) where {D,R,T}
    D::Int
    @assert D >= 0
    R::Int
    @assert 0 <= R <= D
    @assert R == D              # we could relax this
    regular_simplex1(D, T)
end

end
