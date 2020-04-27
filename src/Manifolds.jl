using ComputedFieldTypes
using SparseArrays
using StaticArrays



bitsign(b::Bool) = b ? -1 : 1
bitsign(i::Integer) = bitsign(isodd(i))



"""
    sort_perm

Sort and count permutations.
"""
function sort_perm(xs::SVector{D, T}) where {D, T}
    rs = xs
    s = false
    for i in 1:D
        imin = i
        for j in i+1:D
            xs[j] < xs[imin] && (imin = j)
        end
        rs = setindex(rs, xs[imin], i)
        xs = setindex(xs, xs[i], imin)
        s = xor(s, isodd(i - imin))
    end
    @assert issorted(rs)
    rs, s
end



# The name "Simplex" is taken by Grassmann; we thus use "Manifold
# Simplex (MSimplex)" instead
export MSimplex
@computed struct MSimplex{D, T}
    vertices::fulltype(SVector{D+1, T})
    signbit::Bool

    function MSimplex{D, T}(vertices::SVector{D1, T},
                            signbit::Bool=false) where {D, T, D1}
        @assert D1 == D+1
        v, s = sort_perm(vertices)
        new{D, T}(v, xor(signbit, s))
    end
    function MSimplex(vertices::SVector{D1, T},
                      signbit::Bool=false) where {D1, T}
        D = D1-1
        MSimplex{D, T}(vertices, signbit)
    end
end

export invariant
function invariant(s::MSimplex)::Bool
    issorted(s.vertices)
end

function Base.show(io::IO, s::MSimplex)
    print(io, "($(s.vertices); $(bitsign(s.signbit)))")
end

Base.:(==)(s::S, t::S) where {S<:MSimplex} =
    s.vertices == t.vertices && s.signbit == t.signbit
function Base.isless(s::S, t::S) where {S<:MSimplex}
    isless(s.vertices, t.vertices) && return true
    isless(t.vertices, s.vertices) && return false
    isless(s.signbit, t.signbit)
end

export dim
dim(::Type{S}) where {S<:MSimplex} = length(S) - 1
dim(::S) where {S<:MSimplex} = dim(S)

Base.getindex(s::MSimplex, i) = s.vertices[i]
Base.length(::Type{<:MSimplex{D}}) where {D} = D + 1
Base.length(::S) where {S<:MSimplex} = length(S)



function simplices_type(D::Int, ::Type{T})::Type where {T}
    Tuple{(Vector{fulltype(MSimplex{d, T})} for d in 1:D)...}
end

export Manifold
# """
# Manifold (aka Chain)
# """
@computed struct Manifold{D}
    nvertices::Int
    # vertices (i.e. 0-simplices) are always numbered 1:nvertices and
    # are not stored
    simplices::simplices_type(D, Int)
    # The boundary ∂ of 0-forms vanishes and is not stored
    boundaries::NTuple{D, SparseMatrixCSC{Int8, Int}}

    function Manifold{D}(nvertices::Int,
                         simplices::NTuple{D, Vector{<:MSimplex}},
                         boundaries::NTuple{D, SparseMatrixCSC{Int8, Int}}
                         ) where {D}
        simplices::simplices_type(D, Int)
        mf = new{D}(nvertices, simplices, boundaries)
        @assert invariant(mf)
        mf
    end
    function Manifold(nvertices::Int,
                      simplices::NTuple{D, Vector{<:MSimplex}},
                      boundaries::NTuple{D, SparseMatrixCSC{Int8, Int}}
                      ) where {D}
        Manifold{D}(nvertices, simplices, boundaries)
    end
end
# TODO: Implement also the "cube complex" representation

export invariant
function invariant(mf::Manifold{D})::Bool where {D}
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
        size(boundaries) == (dim(R-1, mf), dim(R, mf)) ||
            (@assert false; return false)
    end

    return true
end

# Comparison

function Base.:(==)(mf1::Manifold{D}, mf2::Manifold{D})::Bool where {D}
    mf1.nvertices == mf2.nvertices || return false
    mf1.simplices == mf2.simplices
end

Base.ndims(::Manifold{D}) where {D} = D

function dim(::Val{R}, mf::Manifold{D})::Int where {R, D}
    R == 0 && return mf.nvertices
    length(mf.simplices[R])
end
function dim(R, mf::Manifold)::Int
    R == 0 && return mf.nvertices
    length(mf.simplices[R])
end

# Convenience constructors

function Manifold(simplices::Vector{MSimplex{D, Int, X}}
                  )::Manifold{D} where {D, X}
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
        return Manifold{D}(nvertices, (), ())
    end

    # Calculate lower-dimensional simplices
    # See arXiv:1103.3076, section 7
    faces = fulltype(MSimplex{D-1, Int})[]
    boundaries1 = Tuple{fulltype(MSimplex{D-1}), Int}[]
    for (i,s) in enumerate(simplices)
        for a in 1:D+1
            # Leave out vertex a
            v1 = SVector{D}(ntuple(b -> s[b + (b>=a)], D))
            s1 = xor(s.signbit, isodd(a-1))
            face = MSimplex{D-1, Int}(v1)
            # face = MSimplex{D-1, Int}(face.vertices, false)
            boundary1 = (MSimplex{D-1, Int}(v1, s1), i)
            push!(faces, face)
            push!(boundaries1, boundary1)
        end
    end
    sort!(faces)
    unique!(faces)
    mf1 = Manifold(faces)

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

    Manifold{D}(nvertices,
                tuple(mf1.simplices..., simplices),
                tuple(mf1.boundaries..., boundaries))
end

function Manifold(simplices::Vector{<:SVector{D1, Int}}
                  )::Manifold{D1-1} where {D1}
    D = D1-1
    Manifold(fulltype(MSimplex{D, Int})[MSimplex{D, Int}(s) for s in simplices])
end

function Manifold(::Val{D})::Manifold{D} where {D}
    Manifold(fulltype(MSimplex{D, Int})[])
end

function Manifold(simplex::MSimplex{D, Int})::Manifold{D} where {D}
    Manifold([simplex])
end

# Boundary

export boundary
function boundary(::Val{R}, mf::Manifold{D}) where {R, D}
    @assert 0 < R <= D
    return mf.boundary[R]
end
