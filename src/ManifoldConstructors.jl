module ManifoldConstructors

using LinearAlgebra
using Random
using SparseArrays
using StaticArrays

using ..Manifolds
using ..Meshing
using ..SparseOps
using ..ZeroOrOne

################################################################################

export empty_manifold
"""
The empty manifold
"""
function empty_manifold(::Val{D}, ::Type{S}; options...) where {D,S}
    return Manifold("empty manifold", zero(SparseOp{0,D,One}, 0, 0),
                    IDVector{0,SVector{D,S}}(), IDVector{0,S}(); options...)
end

################################################################################

export simplex_manifold
"""
Manifold with one standard simplex
"""
function simplex_manifold(::Val{D}, ::Type{S}; options...) where {D,S}
    nvertices = D + 1
    coords = IDVector{0}(regular_simplex(Val(D), S))
    weights = IDVector{0}(zeros(S, length(coords)))
    simplices = MakeSparse{One}(nvertices, 1)
    for i in 1:nvertices
        simplices[i, 1] = One()
    end
    simplices = SparseOp{0,D,One}(sparse(simplices))
    return Manifold("simplex manifold", simplices, coords, weights; options...)
end

"""
Generate the coordinate positions for a regular D-simplex with edge
length 1.

The algorithm proceeds recursively. A 0-simplex is a point. A
D-simplex is a (D-1)-simplex that is shifted down along the new axis,
plus a new point on the new axis.
"""
function regular_simplex(::Val{D}, ::Type{S}) where {D,S}
    @assert D ≥ 0
    N = D + 1
    s = Array{SVector{D,S}}(undef, N)
    if D > 0
        s0 = regular_simplex(Val(D - 1), S)
        # Choose height so that edge length is 1
        if D == 1
            z = S(1)
        else
            z0 = sqrt(1 - sum(s0[1] .^ 2))
            if S <: Rational
                z = rationalize(typeof(zero(S).den), z0; tol=sqrt(eps(z0)))
            else
                z = z0::S
            end
        end
        z0 = -z / (D + 1)
        for n in 1:(N - 1)
            s[n] = SVector{D,S}(s0[n]..., z0)
        end
        s[N] = SVector{D,S}(zero(SVector{D - 1,S})..., z + z0)
    end
    return s
end

################################################################################

export orthogonal_simplex_manifold
"""
Manifold with one orthogonal simplex
"""
function orthogonal_simplex_manifold(::Val{D}, ::Type{S};
                                     options...) where {D,S}
    N = D + 1
    coords, weights = orthogonal_simplex(Val(D), S)
    # Ignore the suggested weights (why?)
    weights = zeros(S, length(coords))
    coords = IDVector{0}(coords)
    weights = IDVector{0}(weights)
    simplices = MakeSparse{One}(N, 1)
    for i in 1:N
        simplices[i, 1] = One()
    end
    simplices = SparseOp{0,D,One}(sparse(simplices))
    return Manifold("orthogonal simplex manifold", simplices, coords, weights;
                    options...)
end

"""
Generate the coordinate positions for an orthogonal D-simplex with
edge lengths 1.
"""
function orthogonal_simplex(::Val{D}, ::Type{S}) where {D,S}
    @assert D ≥ 0
    s = zeros(SVector{D,S}, D + 1)
    w = zeros(S, D + 1)
    w[1] = Dict(0 => 0, 1 => 0, 2 => -S(1) / 2,
                # 2 => -S(1) / 8,
                3 => -S(1) / 2, 4 => -S(2) / 3, 5 => -S(2) / 3)[D]
    w[1] = -1 + S(1) / D
    for d in 1:D
        s[d + 1] = setindex(zero(SVector{D,S}), 1, d)
    end
    return s, w
end

################################################################################

export hypercube_manifold
"""
Standard tesselation of a hypercube
"""
function hypercube_manifold(::Val{D}, ::Type{S}; options...) where {D,S}
    @assert D ≥ 0
    N = D + 1

    # Find simplices
    simplices = SVector{N,Int}[]
    corner = zero(SVector{D,Bool})
    vertices = [corner2vertex(corner)]
    next_corner!(simplices, vertices, corner)
    nsimplices = length(simplices)
    @assert nsimplices == factorial(D)

    # Set up coordinates
    coords = IDVector{0,SVector{D,S}}()
    imin = CartesianIndex(ntuple(d -> 0, D))
    imax = CartesianIndex(ntuple(d -> 1, D))
    for i in imin:imax
        push!(coords, SVector{D,S}(i.I))
    end
    nvertices = length(coords)
    @assert nvertices == 2^D
    weights = IDVector{0}(zeros(S, nvertices))

    simplices′ = MakeSparse{One}(nvertices, nsimplices)
    for (j, sj) in enumerate(simplices)
        for i in sj
            simplices′[Int(i), Int(j)] = One()
        end
    end
    simplices = SparseOp{0,D,One}(sparse(simplices′))

    return Manifold("hypercube manifold", simplices, coords, weights;
                    options...)
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

################################################################################

export large_hypercube_manifold
"""
Triangulate a large hypercube by splitting it into smaller hypercubes
"""
function large_hypercube_manifold(::Val{D}, ::Type{S}; nelts::NTuple{D,Int},
                                  torus::Bool=false, options...) where {D,S}
    @assert D ≥ 0
    N = D + 1

    @assert all(nelts .≥ 0)
    # We could relax this
    @assert all(nelts .% 2 .== 0)

    symmetric_hypercube = create_hypercube(Val(D))
    for dir in 1:D
        append!(symmetric_hypercube, mirror_hypercube(symmetric_hypercube, dir))
    end
    @assert length(symmetric_hypercube) == factorial(D) * 2^D

    stride = SVector{D,Int}(d == 1 ? 1 : prod(nelts[1:(d - 1)] .+ 1)
                            for d in 1:D)
    function vertex2ind(i::SVector{D,Int})
        if !torus
            return 1 + sum(i .* stride)
        else
            return 1 + sum(i .% nelts .* stride)
        end
    end

    nsimplices = factorial(D) * (D == 0 ? 1 : prod(nelts))
    simplices = Array{SVector{N,Int}}(undef, nsimplices)
    i = 0
    for elt in CartesianIndices(nelts .÷ 2)
        offset = 2 * (SVector{D,Int}(elt.I) .- 1)
        for s in symmetric_hypercube
            simplices[i += 1] = map(v -> vertex2ind(v .+ offset), s)
        end
    end
    @assert i == length(simplices)

    nvertices = D == 0 ? 1 : prod(nelts .+ 1)
    coords = Array{SVector{D,S}}(undef, nvertices)
    i = 0
    for elt in CartesianIndices(nelts .+ 1)
        coords[i += 1] = (SVector{D,S}(elt.I) .- 1) ./ nelts
    end
    @assert i == length(coords)
    coords = IDVector{0}(coords)

    # fact = SVector{D,S}(a == D ? 1 / S(2) : 1 for a in 1:D)
    # map!(x -> fact .* x, coords, coords)

    weights = IDVector{0}(zeros(S, nvertices))

    simplices′ = MakeSparse{One}(nvertices, nsimplices)
    for (j, sj) in enumerate(simplices)
        for i in sj
            simplices′[Int(i), Int(j)] = One()
        end
    end
    simplices = SparseOp{0,D,One}(sparse(simplices′))

    return Manifold("large hypercube manifold", simplices, coords, weights;
                    options...)
end

function create_hypercube(::Val{D}) where {D}
    @assert D ≥ 0
    N = D + 1

    # Find simplices
    simplices = SVector{N,SVector{D,Int}}[]
    corner = zero(SVector{D,Bool})
    vertices = SVector{D,Int}[corner]
    next_corner!(simplices, vertices, corner)
    nsimplices = length(simplices)
    @assert nsimplices == factorial(D)

    # Ensure we have all 2^D vertices
    cvertices = SVector{D,Int}[]
    for si in simplices
        for j in si
            push!(cvertices, j)
        end
    end
    unique!(sort!(cvertices))
    @assert length(cvertices) == 2^D

    return simplices
end

"""
- Accumulate the simplices in `simplices`.
- `vertices` is the current set of vertices as we sweep from the
  origin to diagonally opposide vertex.
- `corner` is the current corner.
"""
function next_corner!(simplices::Vector{SVector{N,SVector{D,Int}}},
                      vertices::Vector{SVector{D,Int}},
                      corner::SVector{D,Bool}) where {N,D}
    @assert N == D + 1
    if D == 0
        @assert length(vertices) == 1
    end
    @assert sum(corner) == length(vertices) - 1
    if length(vertices) == D + 1
        # We have all vertices; build the simplex
        push!(simplices, SVector{N}(vertices))
        return
    end
    # Loop over all neighbouring corners
    for d in 1:D
        if !corner[d]
            new_corner = setindex(corner, true, d)
            new_vertex = SVector{D,Int}(new_corner)
            new_vertices = copy(vertices)
            push!(new_vertices, new_vertex)
            next_corner!(simplices, new_vertices, new_corner)
        end
    end
    return
end

function mirror_hypercube(simplices::Vector{SVector{N,SVector{D,Int}}},
                          dir::Int) where {D,N}
    @assert D ≥ 0
    @assert N == D + 1
    @assert 1 ≤ dir ≤ D

    flip = SVector{D,Int}(d == dir ? -1 : 1 for d in 1:D)
    offset = SVector{D,Int}(d == dir ? 2 : 0 for d in 1:D)
    simplices′ = map(s -> map(v -> flip .* v .+ offset, s), simplices)

    return simplices′
end

function shift_hypercube(simplices::Vector{SVector{N,SVector{D,Int}}},
                         offset::SVector{D,Int}) where {D,N}
    @assert D ≥ 0
    @assert N == D + 1

    simplices′ = map(s -> map(v -> v .+ offset, s), simplices)

    return simplices′
end

################################################################################

export delaunay_hypercube_manifold
"""
Delaunay triangulation of a hypercube
"""
function delaunay_hypercube_manifold(::Val{D}, ::Type{S};
                                     options...) where {D,S}
    @assert D ≥ 0
    N = D + 1

    # Set up coordinates
    coords = IDVector{0,SVector{D,S}}()
    imin = CartesianIndex(ntuple(d -> 0, D))
    imax = CartesianIndex(ntuple(d -> 1, D))
    for i in imin:imax
        # Re-map coordinates to avoid round-off errors
        push!(coords, SVector{D,S}(i.I) .+ S(5) / 2)
    end
    nvertices = length(coords)
    @assert nvertices == 2^D
    weights = IDVector{0}(zeros(S, length(coords)))

    simplices = delaunay_mesh(coords)

    return Manifold("delaunay hypercube manifold", simplices, coords, weights;
                    options...)
end

################################################################################

export large_delaunay_hypercube_manifold
"""
Delaunay triangulation of a large hypercube
"""
function large_delaunay_hypercube_manifold(::Val{D}, ::Type{S},
                                           n::Union{Nothing,Int}=nothing;
                                           options...) where {D,S}
    @assert D ≥ 0
    N = D + 1

    if n ≡ nothing
        ns = Dict{Int,Int}(0 => 1, 1 => 1024, 2 => 32, 3 => 16, 4 => 4, 5 => 2)
        n = ns[D]
    end
    # All n should be powers of 2 to avoid round-off below
    @assert ispow2(n)

    rng = MersenneTwister(1)

    # Set up coordinates
    coords = IDVector{0,SVector{D,S}}()
    # coords′ = SVector{D,S}[]
    imin = CartesianIndex(ntuple(d -> 0, D))
    imax = CartesianIndex(ntuple(d -> n, D))
    for i in imin:imax

        # # We also set up slightly different coordinates to help the
        # # Delaunay triangulation
        # # 1. Perturb points in the interior randomly to avoid
        # #    degeneracies
        # # 2. Make points on the boundary bulge outwards to avoid
        # #    degeneracies
        # 
        # x = SVector{D,S}(i[d] for d in 1:D) / n
        # 
        # # # Avoid floating-point round-off errors
        # # x = x .+ S(5) / 2
        # 
        # # Add noise
        # isbnd(d) = i[d] ∈ (0, n)
        # x += SVector{D,S}(isbnd(d) ? 0 : (2 * rand(rng, S) - 1) / (256 * n)
        #                   for d in 1:D)
        # 
        # # These are the coordinates we want
        # push!(coords, x)
        # 
        # # x′ = x
        # # 
        # # # Barrel transformation
        # # r = norm(2 * x′ .- 1)
        # # x′ = ((2 * x′ .- 1) * (1 - r / (256^3 * n)) .+ 1) / 2
        # # 
        # # push!(coords′, x′)

        # Stagger every second point, except boundary points
        x = SVector{D,S}(i[d] for d in 1:D) / n
        for d in 1:D
            for d′ in 1:(d - 1)
                i[d′] ∈ (0, n) && continue
                di = (S(1) / 2) / sqrt(S(d - 1))
                x1 = x[d′]
                x1 = x1 * n / (n + di)
                if isodd(i[d])
                    x1 = x1 + di / (n + di)
                end
                x = Base.setindex(x, x1, d′)
            end
        end

        push!(coords, x)
    end
    # @assert length(coords′) == length(coords)
    nvertices = length(coords)
    @assert nvertices == (n + 1)^D
    weights = IDVector{0}(zeros(S, length(coords)))

    simplices = delaunay_mesh(coords)

    return Manifold("large delaunay hypercube manifold", simplices, coords,
                    weights; options...)
end

################################################################################

export random_hypercube_manifold
"""
Hypercube manifold with random vertices
"""
function random_hypercube_manifold(xmin::SVector{D,S}, xmax::SVector{D,S},
                                   npoints_per_dim::Int; options...) where {D,S}
    rng = MersenneTwister(1)

    function coord(f, d)
        f == -1 && return xmin[d]
        f == +1 && return xmax[d]
        local n = 2^20
        local i = rand(rng, 1:(n - 1))
        local x = (n - i) / S(n) * xmin[d] + i / S(n) * xmax[d]
        return x
    end

    coords = IDVector{0}(Vector{SVector{D,S}}())
    for face in
        CartesianIndex(ntuple(d -> -1, D)):CartesianIndex(ntuple(d -> 1, D))
        dim = count(==(0), face[d] for d in 1:D)
        for i in 1:(npoints_per_dim^dim)
            x = SVector{D,S}(coord(face[d], d) for d in 1:D)
            push!(coords, x)
        end
    end
    weights = IDVector{0}(zeros(S, length(coords)))

    simplices = delaunay_mesh(coords)

    return Manifold("random hypercube manifold", simplices, coords, weights;
                    options...)
end

################################################################################

export refined_manifold
"""
Refined manifold
"""
function refined_manifold(mfd0::Manifold{D,C,S}; options...) where {D,C,S}
    D == 0 && return mfd0
    coords = refine_coords(get_lookup(mfd0, 0, 1), get_coords(mfd0))
    weights = IDVector{0}(zeros(S, length(coords)))
    simplices = delaunay_mesh(coords)
    mfd = Manifold("refined $(mfd0.name)", simplices, coords, weights;
                   options...)
    return mfd
end

################################################################################

export refined_simplex_manifold
"""
Refined simplex manifold
"""
function refined_simplex_manifold(::Val{D}, ::Type{S}; options...) where {D,S}
    mfd0 = simplex_manifold(Val(D), S; options...)
    mfd = refined_manifold(mfd0; options...)
    return mfd
end

################################################################################

export boundary_manifold
"""
Boundar manifold
"""
function boundary_manifold(mfd0::Manifold{D,C,S}; options...) where {D,C,S}
    @assert D > 0
    # Find boundary faces
    B = get_boundaries(mfd0, D)
    volume_weights = IDVector{D}(ones(Int8, nsimplices(mfd0, D)))
    keep_face = B * volume_weights
    keep_face::IDVector{D - 1,Int8}
    @assert all(x -> abs(x) ≤ 1, keep_face)
    faces_old2new = IDVector{D - 1}(zeros(ID{D - 1}, nsimplices(mfd0, D - 1)))
    faces_new2old = IDVector{D - 1,ID{D - 1}}()
    for j in axes(get_simplices(mfd0, D - 1), 2)
        if keep_face[j] ≠ 0
            push!(faces_new2old, j)
            faces_old2new[j] = ID{D - 1}(length(faces_new2old))
        end
    end
    nfaces_new = length(faces_new2old)
    # Find boundary vertices
    F = get_simplices(mfd0, D - 1)
    keep_vertex = IDVector{0}(zeros(Bool, nsimplices(mfd0, 0)))
    for j in faces_new2old
        for i in sparse_column_rows(F, j)
            keep_vertex[i] = true
        end
    end
    vertices_old2new = IDVector{0}(zeros(ID{0}, nsimplices(mfd0, 0)))
    vertices_new2old = IDVector{0,ID{0}}()
    for i in axes(get_simplices(mfd0, 0), 2)
        if keep_vertex[i]
            push!(vertices_new2old, i)
            vertices_old2new[i] = ID{0}(length(vertices_new2old))
        end
    end
    nvertices_new = length(vertices_new2old)
    # Define new manifold
    name = "boundary $(mfd0.name)"
    simplices = MakeSparse{One}(nvertices_new, nfaces_new)
    for jnew in axes(faces_new2old, 1)
        jold = faces_new2old[jnew]
        for iold in sparse_column_rows(F, jold)
            inew = vertices_old2new[iold]
            simplices[Int(inew), Int(jnew)] = One()
        end
    end
    simplices = SparseOp{0,D - 1,One}(sparse(simplices))
    coords = get_coords(mfd0)[vertices_new2old]
    weights = get_weights(mfd0)[vertices_new2old]
    mfd = Manifold(name, simplices, coords, weights; options...)
    return mfd
end

################################################################################

export boundary_simplex_manifold
"""
Boundary simplex manifold
"""
function boundary_simplex_manifold(::Val{D}, ::Type{S}; options...) where {D,S}
    mfd0 = simplex_manifold(Val(D + 1), S; options...)
    mfd = boundary_manifold(mfd0; options...)
    return mfd
end

end
