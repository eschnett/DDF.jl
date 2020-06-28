module Topologies

using LinearAlgebra
using SparseArrays
using StaticArrays

using ..Defs



# TODO: do we really need this type? it only makes sense if the sign
# bit is used often. check this, after the algorithms work correctly.
# add golden tests for derivatives on topologies (simplex, hypercube,
# surface of simplex/hypercube, etc.)
export Simplex
struct Simplex{N, T}
    vertices::SVector{N, T}
    signbit::Bool

    function Simplex{N, T}(vertices::SVector{N, T},
                           signbit::Bool=false) where {N, T}
        N::Int
        T::Type
        v, s = sort_perm(vertices)
        new{N, T}(v, signbit ⊻ isodd(s))
    end
    function Simplex(vertices::SVector{N, T},
                     signbit::Bool=false) where {N, T}
        Simplex{N, T}(vertices, signbit)
    end
end

function Defs.invariant(s::Simplex)::Bool
    issorted(s.vertices)
end

function Base.show(io::IO, s::Simplex)
    print(io, "($(bitsign(s.signbit)))$(s.vertices)")
end

Base.:(==)(s::S, t::S) where {S<:Simplex} =
    s.vertices == t.vertices && s.signbit == t.signbit
function Base.isless(s::S, t::S) where {S<:Simplex}
    isless(s.vertices, t.vertices) && return true
    isless(t.vertices, s.vertices) && return false
    isless(s.signbit, t.signbit)
end

Base.ndims(::Type{S}) where {S<:Simplex} = length(S) - 1
Base.ndims(::S) where {S<:Simplex} = ndims(S)

Base.getindex(s::Simplex, i) = s.vertices[i]
Base.length(::Type{<:Simplex{N}}) where {N} = N
Base.length(::S) where {S<:Simplex} = length(S)



const Simplices{N} = Vector{Simplex{N, Int}}

# TODO: Introduce type for operators

export Topology
"""
Topology (a set of directed graphs)
"""
struct Topology{D}
    nvertices::Int
    # vertices (i.e. 0-simplices) are always numbered 1:nvertices and
    # don't need to be stored
    # simplices[R]::Vector{Simplex{R+1, Int}}
    simplices::Dict{Int, Simplices}
    # The boundary ∂ of 0-forms vanishes and is not stored
    boundaries::Dict{Int, SparseMatrixCSC{Int8, Int}}

    function Topology{D}(nvertices::Int,
                         simplices::Dict{Int, Simplices},
                         boundaries::Dict{Int, SparseMatrixCSC{Int8, Int}}
                         ) where {D}
        D::Int
        @assert D >= 0
        @assert isempty(symdiff(keys(simplices), 0:D))
        @assert isempty(symdiff(keys(boundaries), 1:D))
        topo = new{D}(nvertices, simplices, boundaries)
        @assert invariant(topo)
        topo
    end
    function Topology(nvertices::Int,
                      simplices::Dict{Int, Simplices},
                      boundaries::Dict{Int, SparseMatrixCSC{Int8, Int}}
                      ) where {D}
        Topology{D}(nvertices, simplices, boundaries)
    end
end
# TODO: Implement also the "cube complex" representation

function Defs.invariant(topo::Topology{D})::Bool where {D}
    D >= 0 || (@assert false; return false)

    topo.nvertices >= 0 || (@assert false; return false)

    isempty(symdiff(keys(topo.simplices), 0:D)) || (@assert false; return false)
    length(topo.simplices[0]) == topo.nvertices || (@assert false; return false)

    for R in 0:D
        simplices = topo.simplices[R]
        for i in 1:length(simplices)
            s = simplices[i]
            for d in 1:R+1
                1 <= s[d] <= topo.nvertices || (@assert false; return false)
            end
            for d in 2:R+1
                s[d] > s[d-1] || (@assert false; return false)
            end
            if i > 1
                s > simplices[i-1] || (@assert false; return false)
            end
        end
    end

    for R in 1:D
        boundaries = topo.boundaries[R]
        size(boundaries) == (size(R-1, topo), size(R, topo)) ||
            (@assert false; return false)
    end

    return true
end

# Comparison

function Base.:(==)(topo1::Topology{D}, topo2::Topology{D})::Bool where {D}
    topo1.nvertices == topo2.nvertices || return false
    topo1.simplices == topo2.simplices
end

Base.ndims(::Topology{D}) where {D} = D

Base.size(::Val{R}, topo::Topology{D}) where {R, D} = size(R, topo)
function Base.size(R::Integer, topo::Topology)::Int
    @assert 0 <= R <= ndims(topo)
    length(topo.simplices[R])
end

# Constructors

function Topology(simplices::Vector{Simplex{N, Int}}
                  )::Topology{N-1} where {N}
    D = N-1
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
        return Topology{D}(nvertices,
                           Dict{Int, Simplices}(0 => simplices),
                           Dict{Int, SparseMatrixCSC{Int8, Int}}())
    end

    # Calculate lower-dimensional simplices
    # See arXiv:1103.3076v2 [cs.NA], section 7
    faces = Simplex{N-1, Int}[]
    boundaries1 = Tuple{Simplex{N-1}, Int}[]
    for (i,s) in enumerate(simplices)
        for a in 1:N
            # Leave out vertex a
            v1 = SVector{N-1}(ntuple(b -> s[b + (b>=a)], N-1))
            s1 = xor(s.signbit, isodd(a-1))
            face = Simplex{N-1, Int}(v1)
            # face = Simplex{N-1, Int}(face.vertices, false)
            boundary1 = (Simplex{N-1, Int}(v1, s1), i)
            push!(faces, face)
            push!(boundaries1, boundary1)
        end
    end
    sort!(faces)
    unique!(faces)
    topo1 = Topology(faces)

    sort!(boundaries1)
    @assert allunique(boundaries1)
    I = Int[]
    J = Int[]
    V = Int8[]
    i = 0
    lastv = nothing
    for (s,j) in boundaries1
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
    topo1.boundaries[D] = boundaries
    Topology{D}(nvertices, topo1.simplices, topo1.boundaries)
end

function Topology(simplices::Vector{SVector{N, Int}}
                   )::Topology{N-1} where {N}
    Topology([Simplex{N, Int}(s) for s in simplices])
end

function Topology(::Val{D})::Topology{D} where {D}
    Topology(Simplex{D+1, Int}[])
end

function Topology(simplex::Simplex{N, Int})::Topology{N-1} where {N}
    Topology(Simplex{N, Int}[simplex])
end



function corner2vertex(c::SVector{D,Bool})::Int where {D}
    D==0 && return 1
    1 + sum(c[d] << (d-1) for d in 1:D)
end

function next_corner!(simplices::Vector{Simplex{N, Int}},
                      vertices::SVector{M, Int},
                      corner::SVector{D, Bool})::Nothing where {N, D, M}
    @assert N == D+1
    if D == 0
        @assert M == 1
    else
        @assert sum(Int(corner[d]) for d in 1:D) == M - 1
    end
    if M == D+1
        # We have all vertices; build the simplex
        push!(simplices, Simplex(vertices))
        return
    end
    # Loop over all neighbouring corners
    for d in 1:D
        if !corner[d]
            new_corner = setindex(corner, true, d)
            new_vertex = corner2vertex(new_corner)
            new_vertices = SVector{M+1,Int}(vertices..., new_vertex)
            next_corner!(simplices, new_vertices, new_corner)
        end
    end
    nothing
end

export hypercube_manifold
function hypercube_manifold(::Val{D}) where {D}
    simplices = Simplex{D+1,Int}[]
    corner = zeros(SVector{D,Bool})
    vertex = corner2vertex(corner)
    next_corner!(simplices, SVector{1,Int}(vertex), corner)
    @assert length(simplices) == factorial(D)
    Topology(simplices)
end

end
