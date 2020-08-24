module ManifoldConstructors

using LinearAlgebra
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
function empty_manifold(::Val{D}, ::Type{S}) where {D,S}
    return Manifold("empty manifold", zero(SparseOp{Rank{0},Rank{D},One}, 0, 0),
                    zeros(S, 0, D))
end

################################################################################

export simplex_manifold
"""
Manifold with one standard simplex
"""
function simplex_manifold(::Val{D}, ::Type{S}) where {D,S}
    N = D + 1
    coords = regular_simplex(D, S)
    I = Int[]
    J = Int[]
    V = One[]
    for i in 1:N
        push!(I, i)
        push!(J, 1)
        push!(V, One())
    end
    simplices = SparseOp{Rank{0},Rank{D},One}(sparse(I, J, V, N, 1))
    return Manifold("simplex manifold", simplices, coords)
end

"""
Generate the coordinate positions for a regular D-simplex with edge
length 1.

The algorithm proceeds recursively. A 0-simplex is a point. A
D-simplex is a (D-1)-simplex that is shifted down along the new axis,
plus a new point on the new axis.
"""
function regular_simplex(D::Int, ::Type{S}) where {S}
    @assert D >= 0
    N = D + 1
    s = Array{S}(undef, N, D)
    if D > 0
        s0 = regular_simplex(D - 1, S)
        # Choose height so that edge length is 1
        if D == 1
            z = S(1)
        else
            z0 = sqrt(1 - norm(s0[1, :]))
            if S <: Rational
                z = rationalize(S, z0; tol = eps(z0))
            else
                z = z0::S
            end
        end
        z0 = -z / (D + 1)
        s[1:(N - 1), 1:(D - 1)] .= s0[:, :]
        s[1:(N - 1), D] .= z0
        s[N, 1:(D - 1)] .= 0
        s[N, D] = z + z0
    end
    return s
end

################################################################################

export hypercube_manifold
"""
Standard simplification of a hypercube
"""
function hypercube_manifold(::Val{D}, ::Type{S}) where {D,S}
    @assert D >= 0
    N = D + 1

    # Find simplices
    simplices = SVector{N,Int}[]
    corner = zeros(SVector{D,Bool})
    vertices = [corner2vertex(corner)]
    next_corner!(simplices, vertices, corner)
    nsimplices = length(simplices)
    @assert nsimplices == factorial(D)

    # Set up coordinates
    coords = SVector{D,Int}[]
    imin = CartesianIndex(ntuple(d -> 0, D))
    imax = CartesianIndex(ntuple(d -> 1, D))
    for i in imin:imax
        push!(coords, SVector{D,Int}(i.I))
    end
    nvertices = length(coords)
    @assert nvertices == 2^D

    I = Int[]
    J = Int[]
    V = One[]
    for (j, sj) in enumerate(simplices)
        for i in sj
            push!(I, i)
            push!(J, j)
            push!(V, One())
        end
    end

    simplices = SparseOp{Rank{0},Rank{D},One}(sparse(I, J, V, nvertices,
                                                     nsimplices))
    coords = S[coords[i][d] for i in 1:nvertices, d in 1:D]

    return Manifold("hypercube manifold", simplices, coords)
end

"""
- Accumulate the simplices in `simplices`.
- `vertices` is the current set of vertices as we sweep from the
  origin to diagonally opposide vertex.
- `corner` is the current corner.
"""
function next_corner!(simplices::Vector{SVector{N,Int}}, vertices::Vector{Int},
                      corner::SVector{D,Bool})::Nothing where {N,D}
    @assert N == D + 1
    if D == 0
        @assert length(vertices) == 1
    end
    @assert count(corner) == length(vertices) - 1
    if length(vertices) == D + 1
        # We have all vertices; build the simplex
        push!(simplices, SVector{N}(vertices))
        return
    end
    # Loop over all neighbouring corners
    for d in 1:D
        if !corner[d]
            new_corner = setindex(corner, true, d)
            new_vertex = corner2vertex(new_corner)
            new_vertices = [vertices; new_vertex]
            next_corner!(simplices, new_vertices, new_corner)
        end
    end
end

function corner2vertex(c::SVector{D,Bool})::Int where {D}
    D == 0 && return 1
    return 1 + sum(c[d] << (d - 1) for d in 1:D)
end

end
