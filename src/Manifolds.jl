module Manifolds

using LinearAlgebra
using SparseArrays
using StaticArrays

using ..Defs



# The name "Simplex" is taken by Grassmann; we thus use "DSimplex"
# instead
export DSimplex
struct DSimplex{N, T}
    vertices::SVector{N, T}
    signbit::Bool

    function DSimplex{N, T}(vertices::SVector{N, T},
                            signbit::Bool=false) where {N, T}
        N::Int
        T::Type
        v, s = sort_perm(vertices)
        new{N, T}(v, xor(signbit, s))
    end
    function DSimplex(vertices::SVector{N, T},
                      signbit::Bool=false) where {N, T}
        DSimplex{N, T}(vertices, signbit)
    end
end

function Defs.invariant(s::DSimplex)::Bool
    issorted(s.vertices)
end

function Base.show(io::IO, s::DSimplex)
    print(io, "($(s.vertices); $(bitsign(s.signbit)))")
end

Base.:(==)(s::S, t::S) where {S<:DSimplex} =
    s.vertices == t.vertices && s.signbit == t.signbit
function Base.isless(s::S, t::S) where {S<:DSimplex}
    isless(s.vertices, t.vertices) && return true
    isless(t.vertices, s.vertices) && return false
    isless(s.signbit, t.signbit)
end

Base.ndims(::Type{S}) where {S<:DSimplex} = length(S) - 1
Base.ndims(::S) where {S<:DSimplex} = ndims(S)

Base.getindex(s::DSimplex, i) = s.vertices[i]
Base.length(::Type{<:DSimplex{N}}) where {N} = N
Base.length(::S) where {S<:DSimplex} = length(S)



const Simplices{N} = Vector{DSimplex{N,Int}}

# TODO: Introduce type for operators

# The name "Manifold" is taken by Grassmann; we thus use "DManifold"
# instead
export DManifold
"""
DManifold (a set of directed graphs)
"""
struct DManifold{D}
    nvertices::Int
    # vertices (i.e. 0-simplices) are always numbered 1:nvertices and
    # don't need to be stored
    # simplices[R]::Vector{DSimplex{R+1, Int}}
    simplices::Dict{Int, Simplices}
    # The boundary ∂ of 0-forms vanishes and is not stored
    boundaries::Dict{Int, SparseMatrixCSC{Int8, Int}}

    function DManifold{D}(nvertices::Int,
                          simplices::Dict{Int, Simplices},
                          boundaries::Dict{Int, SparseMatrixCSC{Int8, Int}}
                          ) where {D}
        D::Int
        @assert D >= 0
        @assert isempty(symdiff(keys(simplices), 0:D))
        @assert isempty(symdiff(keys(boundaries), 1:D))
        mf = new{D}(nvertices, simplices, boundaries)
        @assert invariant(mf)
        mf
    end
    function DManifold(nvertices::Int,
                       simplices::Dict{Int, Simplices},
                       boundaries::Dict{Int, SparseMatrixCSC{Int8, Int}}
                       ) where {D}
        DManifold{D}(nvertices, simplices, boundaries)
    end
end
# TODO: Implement also the "cube complex" representation

function Defs.invariant(mf::DManifold{D})::Bool where {D}
    D >= 0 || (@assert false; return false)

    mf.nvertices >= 0 || (@assert false; return false)

    isempty(symdiff(keys(mf.simplices), 0:D)) || (@assert false; return false)
    length(mf.simplices[0]) == mf.nvertices || (@assert false; return false)

    for R in 0:D
        simplices = mf.simplices[R]
        for i in 1:length(simplices)
            s = simplices[i]
            for d in 1:R+1
                1 <= s[d] <= mf.nvertices || (@assert false; return false)
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
        boundaries = mf.boundaries[R]
        size(boundaries) == (size(R-1, mf), size(R, mf)) ||
            (@assert false; return false)
    end

    return true
end

# Comparison

function Base.:(==)(mf1::DManifold{D}, mf2::DManifold{D})::Bool where {D}
    mf1.nvertices == mf2.nvertices || return false
    mf1.simplices == mf2.simplices
end

Base.ndims(::DManifold{D}) where {D} = D

Base.size(::Val{R}, mf::DManifold{D}) where {R, D} = size(R, mf)
function Base.size(R::Integer, mf::DManifold)::Int
    @assert 0 <= R <= ndims(mf)
    length(mf.simplices[R])
end

# Constructors

function DManifold(simplices::Vector{DSimplex{N, Int}}
                   )::DManifold{N-1} where {N}
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
        return DManifold{D}(nvertices,
                            Dict{Int, Simplices}(0 => simplices),
                            Dict{Int, SparseMatrixCSC{Int8, Int}}())
    end

    # Calculate lower-dimensional simplices
    # See arXiv:1103.3076v2 [cs.NA], section 7
    faces = DSimplex{N-1, Int}[]
    boundaries1 = Tuple{DSimplex{N-1}, Int}[]
    for (i,s) in enumerate(simplices)
        for a in 1:N
            # Leave out vertex a
            v1 = SVector{N-1}(ntuple(b -> s[b + (b>=a)], N-1))
            s1 = xor(s.signbit, isodd(a-1))
            face = DSimplex{N-1, Int}(v1)
            # face = DSimplex{N-1, Int}(face.vertices, false)
            boundary1 = (DSimplex{N-1, Int}(v1, s1), i)
            push!(faces, face)
            push!(boundaries1, boundary1)
        end
    end
    sort!(faces)
    unique!(faces)
    mf1 = DManifold(faces)

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

    mf1.simplices[D] = simplices
    mf1.boundaries[D] = boundaries
    DManifold{D}(nvertices, mf1.simplices, mf1.boundaries)
end

function DManifold(simplices::Vector{SVector{N, Int}}
                   )::DManifold{N-1} where {N}
    DManifold([DSimplex{N, Int}(s) for s in simplices])
end

function DManifold(::Val{D})::DManifold{D} where {D}
    DManifold(DSimplex{D+1, Int}[])
end

function DManifold(simplex::DSimplex{N, Int})::DManifold{N-1} where {N}
    DManifold(DSimplex{N, Int}[simplex])
end



function corner2vertex(c::SVector{D,Bool})::Int where {D}
    1 + sum(Int, d -> c[d] << (d-1), Val(D))
end

function next_corner!(simplices::Vector{DSimplex{N, Int}},
                      vertices::SVector{M, Int},
                      corner::SVector{D, Bool})::Nothing where {N, D, M}
    @assert N == D+1
    @assert sum(Int, d->Int(corner[d]), Val(D)) == M - 1
    if M == D+1
        # We have all vertices; build the simplex
        push!(simplices, DSimplex(vertices))
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
    simplices = DSimplex{D+1,Int}[]
    corner = sarray(Bool, d->false, Val(D))
    vertex = corner2vertex(corner)
    next_corner!(simplices, SVector{1,Int}(vertex), corner)
    @assert length(simplices) == factorial(D)
    DManifold(simplices)
end

end
