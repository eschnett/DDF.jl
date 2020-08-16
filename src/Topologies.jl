module Topologies

using LinearAlgebra
using OrderedCollections
using SparseArrays
using StaticArrays

using ..Defs

# TODO: Do we really need this type? it only makes sense if the sign
# bit is used often. check this, after the algorithms work correctly.
# add golden tests for derivatives on topologies (simplex, hypercube,
# surface of simplex/hypercube, etc.)
# TODO: Use GeometryBasics.Ngon instead? Use GeometryBasics.Point
# instead of SVector / Form?
export Simplex
struct Simplex{N,T}
    vertices::SVector{N,T}
    signbit::Bool

    function Simplex{N,T}(vertices::SVector{N,T},
                          signbit::Bool = false) where {N,T}
        N::Int
        T::Type
        v, s = sort_perm(vertices)
        return new{N,T}(v, signbit ⊻ isodd(s))
    end
    function Simplex(vertices::SVector{N,T}, signbit::Bool = false) where {N,T}
        return Simplex{N,T}(vertices, signbit)
    end
end

function Defs.invariant(s::Simplex)::Bool
    return issorted(s.vertices)
end

function Base.show(io::IO, s::Simplex)
    sign = s.signbit ? "-" : "+"
    return print(io, "⟨$sign⟩$(s.vertices)")
end

function Base.:(==)(s::S, t::S) where {S<:Simplex}
    return s.vertices == t.vertices && s.signbit == t.signbit
end
function Base.isless(s::S, t::S) where {S<:Simplex}
    isless(s.vertices, t.vertices) && return true
    isless(t.vertices, s.vertices) && return false
    return isless(s.signbit, t.signbit)
end

Base.ndims(::Type{S}) where {S<:Simplex} = length(S) - 1
Base.ndims(::S) where {S<:Simplex} = ndims(S)

Base.getindex(s::Simplex, i) = s.vertices[i]
Base.length(::Type{<:Simplex{N}}) where {N} = N
Base.length(::S) where {S<:Simplex} = length(S)

export Simplices
const Simplices{N} = Vector{Simplex{N,Int}}

# TODO: Introduce type for operators

export Topology
"""
Topology (a set of directed graphs)
"""
struct Topology{D}
    name::String
    nvertices::Int
    # vertices (i.e. 0-simplices) are always numbered 1:nvertices and
    # don't need to be stored
    # simplices[R]::Vector{Simplex{R+1, Int}}
    simplices::Dict{Int,Simplices}
    # map from vertices to containing simplices
    lookup::Dict{Int,SparseMatrixCSC{Nothing,Int}}
    # The boundary ∂ of 0-forms vanishes and is not stored
    boundaries::Dict{Int,SparseMatrixCSC{Int8,Int}}

    function Topology{D}(name::String, nvertices::Int,
                         simplices::Dict{Int,Simplices},
                         lookup::Dict{Int,SparseMatrixCSC{Nothing,Int}},
                         boundaries::Dict{Int,SparseMatrixCSC{Int8,Int}}) where {D}
        D::Int
        @assert D >= 0
        @assert isempty(symdiff(keys(simplices), 0:D))
        @assert isempty(symdiff(keys(lookup), 1:D))
        @assert isempty(symdiff(keys(boundaries), 1:D))
        topo = new{D}(name, nvertices, simplices, lookup, boundaries)
        @assert invariant(topo)
        return topo
    end
    function Topology(name::String, nvertices::Int,
                      simplices::Dict{Int,Simplices},
                      lookup::Dict{Int,SparseMatrixCSC{Nothing,Int}},
                      boundaries::Dict{Int,SparseMatrixCSC{Int8,Int}}) where {D}
        return Topology{D}(name, nvertices, simplices, lookup, boundaries)
    end
end
# TODO: Implement also the "cube complex" representation

function Base.show(io::IO, topo::Topology{D}) where {D}
    println(io)
    println(io, "Topology{$D}(")
    println(io, "    name=$(topo.name)")
    println(io, "    nvertices=$(topo.nvertices)")
    for (d, ss) in sort!(OrderedDict(topo.simplices))
        println(io, "    simplices[$d]=$ss")
    end
    for (d, bs) in sort!(OrderedDict(topo.boundaries))
        print(io, "    boundaries[$d]=$bs")
    end
    return print(io, ")")
end

function Defs.invariant(topo::Topology{D})::Bool where {D}
    D >= 0 || (@assert false; return false)

    topo.nvertices >= 0 || (@assert false; return false)

    isempty(symdiff(keys(topo.simplices), 0:D)) || (@assert false; return false)
    length(topo.simplices[0]) == topo.nvertices || (@assert false; return false)

    for R in 0:D
        simplices = topo.simplices[R]
        for i in 1:length(simplices)
            s = simplices[i]
            for d in 1:(R + 1)
                1 <= s[d] <= topo.nvertices || (@assert false; return false)
            end
            for d in 2:(R + 1)
                s[d] > s[d - 1] || (@assert false; return false)
            end
            if i > 1
                s > simplices[i - 1] || (@assert false; return false)
            end
        end
    end

    for R in 1:D
        lookup = topo.lookup[R]
        size(lookup) == (size(R, topo), size(0, topo)) ||
            (@assert false; return false)
        simplices = topo.simplices[R]
        nnz(lookup) == (R + 1) * length(simplices) ||
            (@assert false; return false)
        rows = rowvals(lookup)
        for i in 1:size(lookup, 2)
            for j0 in nzrange(lookup, i)
                j = rows[j0]
                i in simplices[j].vertices || (@assert false; return false)
            end
        end
    end

    for R in 1:D
        boundaries = topo.boundaries[R]
        size(boundaries) == (size(R - 1, topo), size(R, topo)) ||
            (@assert false; return false)
    end

    return true
end

# Comparison

function Base.:(==)(topo1::Topology{D}, topo2::Topology{D})::Bool where {D}
    topo1.nvertices == topo2.nvertices || return false
    return topo1.simplices == topo2.simplices
end

Base.ndims(::Topology{D}) where {D} = D

Base.size(::Val{R}, topo::Topology{D}) where {R,D} = size(R, topo)
function Base.size(R::Integer, topo::Topology)::Int
    @assert 0 <= R <= ndims(topo)
    return length(topo.simplices[R])
end

# Constructors

function Topology(name::String,
                  simplices::Vector{Simplex{N,Int}})::Topology{N - 1} where {N}
    D = N - 1
    # # Ensure simplex vertices are sorted
    # for s in simplices
    #     for a in 2:N
    #         @assert s[a] > s[a-1]
    #     end
    # end
    # # Ensure simplices are sorted
    # for i in 2:length(simplices)
    #     @assert simplices[i] > simplices[i-1]
    # end
    # Count vertices
    nvertices = 0
    for s in simplices
        for a in 1:N
            nvertices = max(nvertices, s[a])
        end
    end
    # # Ensure all vertices are mentioned (we could omit this check)
    # vertices = falses(nvertices)
    # for s in simplices
    #     for a in 1:N
    #         vertices[s[s]] = true
    #     end
    # end
    # @assert all(vertices)

    simplices = copy(simplices)
    sort!(simplices)
    unique!(simplices)
    if D == 0
        return Topology{D}(name, nvertices, Dict{Int,Simplices}(0 => simplices),
                           Dict{Int,SparseMatrixCSC{Nothing,Int}}(),
                           Dict{Int,SparseMatrixCSC{Int8,Int}}())
    end

    # Calculate lower-dimensional simplices
    # See arXiv:1103.3076v2 [cs.NA], section 7
    faces = Simplex{N - 1,Int}[]
    boundaries1 = Tuple{Simplex{N - 1},Int}[]
    for (i, s) in enumerate(simplices)
        for a in 1:N
            # Leave out vertex a
            v1 = SVector{N - 1}(ntuple(b -> s[b + (b >= a)], N - 1))
            s1 = xor(s.signbit, isodd(a - 1))
            face = Simplex{N - 1,Int}(v1)
            # face = Simplex{N-1, Int}(face.vertices, false)
            boundary1 = (Simplex{N - 1,Int}(v1, s1), i)
            push!(faces, face)
            push!(boundaries1, boundary1)
        end
    end
    sort!(faces)
    unique!(faces)
    topo1 = Topology(name, faces)

    I = Int[]
    J = Int[]
    V = Nothing[]
    for (i, si) in enumerate(simplices)
        for j in si.vertices
            push!(I, i)
            push!(J, j)
            push!(V, nothing)
        end
    end
    lookup = sparse(I, J, V)

    sort!(boundaries1)
    @assert allunique(boundaries1)
    I = Int[]
    J = Int[]
    V = Int8[]
    i = 0
    lastv = nothing
    for (s, j) in boundaries1
        if s.vertices != lastv
            i += 1
            lastv = s.vertices
        end
        push!(I, i)
        push!(J, j)
        push!(V, bitsign(s.signbit))
    end
    @assert i == length(faces)
    boundaries = sparse(I, J, V)

    topo1.simplices[D] = simplices
    topo1.lookup[D] = lookup
    topo1.boundaries[D] = boundaries
    return Topology{D}(name, nvertices, topo1.simplices, topo1.lookup,
                       topo1.boundaries)
end

function Topology(name::String,
                  simplices::Vector{SVector{N,Int}})::Topology{N - 1} where {N}
    return Topology(name, [Simplex{N,Int}(s) for s in simplices])
end

function Topology(::Val{D})::Topology{D} where {D}
    return Topology("D=$D emtpy domain", Simplex{D + 1,Int}[])
end

function Topology(simplex::Simplex{N,Int})::Topology{N - 1} where {N}
    return Topology("D=$(N-1) simplex", Simplex{N,Int}[simplex])
end

function corner2vertex(c::SVector{D,Bool})::Int where {D}
    D == 0 && return 1
    return 1 + sum(c[d] << (d - 1) for d in 1:D)
end

function next_corner!(simplices::Vector{Simplex{N,Int}},
                      vertices::SVector{M,Int},
                      corner::SVector{D,Bool})::Nothing where {N,D,M}
    @assert N == D + 1
    if D == 0
        @assert M == 1
    else
        @assert sum(Int(corner[d]) for d in 1:D) == M - 1
    end
    if M == D + 1
        # We have all vertices; build the simplex
        push!(simplices, Simplex(vertices))
        return
    end
    # Loop over all neighbouring corners
    for d in 1:D
        if !corner[d]
            new_corner = setindex(corner, true, d)
            new_vertex = corner2vertex(new_corner)
            new_vertices = SVector{M + 1,Int}(vertices..., new_vertex)
            next_corner!(simplices, new_vertices, new_corner)
        end
    end
    return nothing
end

export hypercube_manifold
function hypercube_manifold(::Val{D}) where {D}
    simplices = Simplex{D + 1,Int}[]
    corner = zeros(SVector{D,Bool})
    vertex = corner2vertex(corner)
    next_corner!(simplices, SVector{1,Int}(vertex), corner)
    @assert length(simplices) == factorial(D)

    coords = SVector{D,Int}[]
    imin = CartesianIndex(ntuple(d -> 0, D))
    imax = CartesianIndex(ntuple(d -> 1, D))
    for i in imin:imax
        push!(coords, SVector{D,Int}(i.I))
    end

    return Topology("D=$D hypercube", simplices), coords
end

"""
Renumber a list of simplices
"""
function renumber(simplices::Simplices, offset::Int, signbit::Bool = false)
    rn(s) = Simplex(s.vertices .+ offset, xor(s.signbit, signbit))
    return rn.(simplices)
end

"""
Calculate how to compress vertices in a list of simplices
"""
function calc_compress(simplices::Simplices)::NTuple{2,Vector{Int}}
    oldnvertices = maximum(maximum(s.vertices) for s in simplices)
    used = falses(oldnvertices)
    for s in simplices
        used[s.vertices] .= true
    end
    old2new = cumsum(used)
    newnvertices = old2new[end]
    new2old = zeros(Int, newnvertices)
    for n in 1:oldnvertices
        if used[n]
            new2old[old2new[n]] = n
        end
    end
    @assert all(!=(0), new2old)
    @assert old2new[new2old] == 1:newnvertices
    return old2new, new2old
end

# Note: This does not yield a Delaunay triangulation
export double_hypercube_manifold
function double_hypercube_manifold(::Val{D}) where {D}
    # Define geometry for a single hypercube
    topo1, coords1 = hypercube_manifold(Val(D))
    simplices1 = topo1.simplices[D]
    simplices1::Simplices{D + 1}

    # Reflect the hypercube geometry
    simplices = Simplices{D + 1}()
    coords = SVector{D,Int}[]
    imin = CartesianIndex(ntuple(d -> 0, D))
    imax = CartesianIndex(ntuple(d -> 1, D))
    for i in imin:imax
        offset = length(simplices)
        signbit = isodd(sum(i.I))
        append!(simplices, renumber(simplices1, offset, signbit))
        reflect(j) = SVector{D}(i.I[d] == 0 ? j[d] : 2 - j[d] for d in 1:D)
        append!(coords, reflect.(coords1))
    end

    # Create map from coordinates to vertices
    coord2vertex = Dict{SVector{D,Int},Int}()
    for (n, x) in enumerate(coords)
        get!(coord2vertex, x, n)
    end
    # Create map from vertices to canonical vertices
    vertex2vertex = Vector{Int}(undef, length(coords))
    for (n, x) in enumerate(coords)
        vertex2vertex[n] = coord2vertex[coords[n]]
    end
    # Renumber vertices
    simplices = map(s -> Simplex(vertex2vertex[s.vertices], s.signbit),
                    simplices)
    # Compress vertices
    old2new, new2old = calc_compress(simplices)
    oldnvertices = length(old2new)
    newnvertices = length(new2old)
    @assert newnvertices <= oldnvertices
    simplices = map(s -> Simplex(old2new[s.vertices], s.signbit), simplices)
    coords = coords[new2old]
    @assert length(coords) == newnvertices

    return Topology("D=$D double hypercube", simplices), coords
end

end
