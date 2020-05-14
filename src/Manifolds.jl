module Manifolds

using ComputedFieldTypes
using SparseArrays
using StaticArrays

using ..Defs



# The name "Simplex" is taken by Grassmann; we thus use "DSimplex"
# instead
export DSimplex
@computed struct DSimplex{D, T}
    vertices::SVector{D+1, T}
    signbit::Bool

    function DSimplex{D, T}(vertices::SVector{D1, T},
                            signbit::Bool=false) where {D, T, D1}
        D::Int
        T::Type
        @assert D1 == D+1
        v, s = sort_perm(vertices)
        new{D, T}(v, xor(signbit, s))
    end
    function DSimplex(vertices::SVector{D1, T},
                      signbit::Bool=false) where {D1, T}
        D = D1-1
        DSimplex{D, T}(vertices, signbit)
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
Base.length(::Type{<:DSimplex{D}}) where {D} = D + 1
Base.length(::S) where {S<:DSimplex} = length(S)



simplices_type(D::Int, ::Type{T}) where {T} =
    Tuple{(Vector{fulltype(DSimplex{R, T})} for R in 1:D)...}

# The name "Manifold" is taken by Grassmann; we thus use "DManifold"
# instead
export DManifold
# """
# DManifold (aka Chain)
# """
@computed struct DManifold{D}
    nvertices::Int
    # vertices (i.e. 0-simplices) are always numbered 1:nvertices and
    # are not stored
    simplices::simplices_type(D, Int)
    # The boundary ∂ of 0-forms vanishes and is not stored
    boundaries::NTuple{D, SparseMatrixCSC{Int8, Int}}

    function DManifold{D}(nvertices::Int,
                          simplices::NTuple{D, Vector{<:DSimplex}},
                          boundaries::NTuple{D, SparseMatrixCSC{Int8, Int}}
                          ) where {D}
        D::Int
        simplices::simplices_type(D, Int)
        mf = new{D}(nvertices, simplices, boundaries)
        @assert invariant(mf)
        mf
    end
    function DManifold(nvertices::Int,
                       simplices::NTuple{D, Vector{<:DSimplex}},
                       boundaries::NTuple{D, SparseMatrixCSC{Int8, Int}}
                       ) where {D}
        DManifold{D}(nvertices, simplices, boundaries)
    end
end
# TODO: Implement also the "cube complex" representation

function Defs.invariant(mf::DManifold{D})::Bool where {D}
    D >= 0 || (@assert false; return false)

    mf.nvertices >= 0 || (@assert false; return false)

    for R in 1:D
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
    R == 0 && return mf.nvertices
    length(mf.simplices[R])
end

# Constructors

function DManifold(simplices::Vector{<:DSimplex{D, Int}}
                   )::DManifold{D} where {D}
    # # Ensure simplex vertices are sorted
    # for s in simplices
    #     for a in 2:D+1
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
        for a in 1:D+1
            nvertices = max(nvertices, s[a])
        end
    end
    # # Ensure all vertices are mentioned (we could omit this check)
    # vertices = falses(nvertices)
    # for s in simplices
    #     for a in 1:D+1
    #         vertices[s[s]] = true
    #     end
    # end
    # @assert all(vertices)

    simplices = copy(simplices)
    sort!(simplices)
    unique!(simplices)
    if D == 0
        return DManifold{D}(nvertices, (), ())
    end

    # Calculate lower-dimensional simplices
    # See arXiv:1103.3076v2 [cs.NA], section 7
    faces = fulltype(DSimplex{D-1, Int})[]
    boundaries1 = Tuple{fulltype(DSimplex{D-1}), Int}[]
    for (i,s) in enumerate(simplices)
        for a in 1:D+1
            # Leave out vertex a
            v1 = SVector{D}(ntuple(b -> s[b + (b>=a)], D))
            s1 = xor(s.signbit, isodd(a-1))
            face = DSimplex{D-1, Int}(v1)
            # face = DSimplex{D-1, Int}(face.vertices, false)
            boundary1 = (DSimplex{D-1, Int}(v1, s1), i)
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

    DManifold{D}(nvertices,
                 tuple(mf1.simplices..., simplices),
                 tuple(mf1.boundaries..., boundaries))
end

function DManifold(simplices::Vector{<:SVector{D1, Int}}
                   )::DManifold{D1-1} where {D1}
    D = D1-1
    DManifold(fulltype(DSimplex{D, Int})[
        DSimplex{D, Int}(s) for s in simplices])
end

function DManifold(::Val{D})::DManifold{D} where {D}
    DManifold(fulltype(DSimplex{D, Int})[])
end

function DManifold(simplex::DSimplex{D, Int})::DManifold{D} where {D}
    DManifold(fulltype(DSimplex{D, Int})[simplex])
end



function corner2vertex(c::SVector{D,Bool})::Int where {D}
    1 + sum(Int, d -> c[d] << (d-1), Val(D))
end

function next_corner!(simplices::Vector{DSimplex{D, Int, X}},
                      vertices::SVector{N, Int},
                      corner::SVector{D, Bool})::Nothing where {D, N, X}
    @assert sum(Int, d->corner[d], Val(D)) == N - 1
    if N == D+1
        # We have all vertices; build the simplex
        push!(simplices, DSimplex(vertices))
        return
    end
    # Loop over all neighbouring corners
    for d in 1:D
        if !corner[d]
            new_corner = setindex(corner, true, d)
            new_vertex = corner2vertex(new_corner)
            new_vertices = SVector{N+1,Int}(vertices..., new_vertex)
            next_corner!(simplices, new_vertices, new_corner)
        end
    end
    nothing
end

export hypercube_manifold
function hypercube_manifold(::Val{D}) where {D}
    simplices = fulltype(DSimplex{D,Int})[]
    corner = sarray(Bool, d->false, Val(D))
    vertex = corner2vertex(corner)
    next_corner!(simplices, SVector{1,Int}(vertex), corner)
    @assert length(simplices) == factorial(D)
    DManifold(simplices)
end

# Boundary

export boundary
function boundary(::Val{R}, mf::DManifold{D}) where {R, D}
    @assert 0 < R <= D
    return mf.boundary[R]
end

end
