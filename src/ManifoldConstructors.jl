module ManifoldConstructors

using SparseArrays
using StaticArrays

using ..Manifolds
using ..SparseOps
using ..ZeroOrOne

################################################################################

export empty_manifold
"""
The empty manifold
"""
function empty_manifold(::Val{D}) where {D}
    return Manifold("empty manifold", zero(SparseOp{Rank{0},Rank{D},One}, 0, 0))
end

################################################################################

export simplex_manifold
"""
Manifold with one standard simplex
"""
function simplex_manifold(::Val{D}, ::Type{T}) where {D,T}
    N = D + 1
    # s = regular_simplex(D, T)
    I = Int[]
    J = Int[]
    V = One[]
    for i in 1:N
        push!(I, i)
        push!(J, 1)
        push!(V, One())
    end
    simplices = SparseOp{Rank{0},Rank{D},One}(sparse(I, J, V, N, 1))
    return Manifold("simplex manifold", simplices)
end

"""
Generate the coordinate positions for a regular D-simplex with edge
length 1.

The algorithm proceeds recursively. A 0-simplex is a point. A
D-simplex is a (D-1)-simplex that is shifted down along the new axis,
plus a new point on the new axis.
"""
function regular_simplex(::Val{D}, ::Type{T}) where {D,T}
    D::Int
    @assert D >= 0
    N = D + 1
    if D == 0
        s = SVector{N}((SVector{D,T}(),))
    else
        s0 = regular_simplex(Val(D - 1), T)
        # Choose height so that edge length is 1
        if D == 1
            z = T(1)
        else
            z = sqrt(1 - abs2(s0[1]))
        end
        z0 = -z / (D + 1)
        s = SVector{N}((map(x -> SVector{D}(x..., z0), s0)...,
                        SVector{D}(zero(SVector{D - 1,T})..., z + z0)))
    end
    return s::SVector{N,SVector{D,T}}
end

function regular_simplex(D::Int, ::Type{T}) where {T}
    @assert D >= 0
    N = D + 1
    s = Array{T}(undef, N, D)
    if D == 0
        # do nothing
    else
        s0 = regular_simplex(D - 1, T)
        # Choose height so that edge length is 1
        if D == 1
            z = T(1)
        else
            z = sqrt(1 - norm(s0[1, :]))
        end
        z0 = -z / (D + 1)
        s[1:(N - 1), 1:(D - 1)] .= s0[:, :]
        s[1:(N - 1), D] .= z0
        s[N, 1:(D - 1)] .= 0
        s[N, D] = z + z0
    end
    return s
end

end
