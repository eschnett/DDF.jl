module Manifolds

using DifferentialForms: bitsign
using SparseArrays
using StaticArrays

using ..Defs
using ..SparseOps
using ..ZeroOrOne

export Rank
struct Rank{N} end

const OpDict{K,T} = Dict{K,SparseOp{<:Rank,<:Rank,T}} where {K,T}

export Manifold
"""
A discrete mfdfold 
"""
struct Manifold{D,T}
    name::String

    # If `simplices[R][i,j]` is present, then vertex `i` is part of
    # the `R`-simplex `j`. `R ∈ 0:D`. We could omit `R=0`.
    simplices::OpDict{Int,One}

    # The boundary ∂ of `0`-forms vanishes and is not stored. If
    # `boundaries[R][i,j] = s`, then `R`-simplex `j` has
    # `(R-1)`-simplex `i` as boundary with orientation `s`. `R ∈ 1:D`.
    boundaries::OpDict{Int,Int8}

    # Lookup tables from `Ri`-simplices to `Rj`-simplices for `Ri<Rj`.
    # If `lookup[(Ri,Rj)][i,j]` is present, then `Rj`-simplex `j`
    # contains `Ri`-simplex `i`. `Ri ∈ 1:D, Rj ∈ Ri+1:D`. We could
    # omit `Rj=Ri+1`.
    lookup::OpDict{Tuple{Int,Int},One}

    function Manifold{D}(name::String, simplices::OpDict{Int,One},
                         boundaries::OpDict{Int,Int8},
                         lookup::OpDict{Tuple{Int,Int},One}) where {D}
        D::Int
        @assert D >= 0
        @assert Set(keys(simplices)) == Set(0:D)
        @assert Set(keys(boundaries)) == Set(1:D)
        @assert Set(keys(lookup)) ==
                Set((Ri, Rj) for Ri in 1:D for Rj in (Ri + 1):D)
        mfd = new{D,Rational}(name, simplices, boundaries, lookup)
        @assert invariant(mfd)
        return mfd
    end
    function Manifold(name::String, simplices::OpDict{Int,One},
                      boundaries::OpDict{Int,Int8},
                      lookup::OpDict{Tuple{Int,Int},One})
        D = maximum(keys(simplices))
        return Manifold{D}(name, simplices, boundaries, lookup)
    end
end
# TODO: Implement also a "cube complex" representation

################################################################################

function Base.show(io::IO, mfd::Manifold{D}) where {D}
    println(io)
    println(io, "Manifold{$D}(")
    println(io, "    name=$(mfd.name)")
    for R in 0:D
        simplices = mfd.simplices[R]
        println(io, "    simplices[$R]=$simplices")
    end
    for R in 1:D
        boundaries = mfd.boundaries[R]
        print(io, "    boundaries[$R]=$boundaries")
    end
    return print(io, ")")
end

function Defs.invariant(mfd::Manifold{D})::Bool where {D}
    D >= 0 || (@assert false; return false)

    # Check simplices
    Set(keys(mfd.simplices)) == Set(0:D) || (@assert false; return false)
    for R in 0:D
        simplices = mfd.simplices[R]::SparseOp{Rank{0},Rank{R},One}
        size(simplices) == (nsimplices(mfd, 0), nsimplices(mfd, R)) ||
            (@assert false; return false)
        for j in 1:size(simplices, 2)
            sj = sparse_column_rows(simplices, j)
            length(sj) == R + 1 || (@assert false; return false)
        end
    end

    # Check boundaries
    Set(keys(mfd.boundaries)) == Set(1:D) || (@assert false; return false)
    for R in 1:D
        boundaries = mfd.boundaries[R]::SparseOp{Rank{R - 1},Rank{R},Int8}
        size(boundaries) == (nsimplices(mfd, R - 1), nsimplices(mfd, R)) ||
            (@assert false; return false)
        for j in 1:size(boundaries, 2) # R-simplex
            vj = sparse_column_rows(mfd.simplices[R], j)
            sj = sparse_column(boundaries, j)
            for (i, p) in sj    # (R-1)-simplex
                si = sparse_column_rows(mfd.simplices[R - 1], i)
                for k in si     # vertices
                    k ∈ vj || (@assert false; return false)
                end
                abs(p) == 1 || (@assert false; return false)
            end
        end
    end

    # Check lookup tables
    Set(keys(mfd.lookup)) == Set((Ri, Rj) for Ri in 1:D for Rj in (Ri + 1):D) ||
        (@assert false; return false)
    for Ri in 1:D, Rj in (Ri + 1):D
        lookup = mfd.lookup[(Ri, Rj)]::SparseOp{Rank{Ri},Rank{Rj},One}
        size(lookup) == (nsimplices(mfd, Ri), nsimplices(mfd, Rj)) ||
            (@assert false; return false)
        for j in 1:size(lookup, 2) # Rj-simplex
            vj = sparse_column_rows(mfd.simplices[Rj], j)
            length(vj) == Rj + 1 || (@assert false; return false)
            sj = sparse_column_rows(lookup, j)
            length(sj) == binomial(Rj + 1, Ri + 1) ||
                (@assert false; return false)
            for i in sj         # Ri-simplex
                si = sparse_column_rows(mfd.simplices[Ri], i)
                for k in si
                    k ∈ vj || (@assert false; return false)
                end
            end
        end
    end

    return true
end

# Comparison

function Base.:(==)(mfd1::Manifold{D}, mfd2::Manifold{D})::Bool where {D}
    return mfd1.simplices == mfd2.simplices
end

Base.ndims(::Manifold{D}) where {D} = D

export nsimplices
nsimplices(mfd::Manifold, ::Val{R}) where {R} = size(mfd.simplices[R], 2)
nsimplices(mfd::Manifold, R::Integer) = size(mfd.simplices[R], 2)

################################################################################

# Outer constructor

struct Face{N}
    vertices::SVector{N,Int}
    parent::Int
    parity::Int8
end

function Manifold{D}(name::String,
                     simplices::SparseOp{Rank{0},Rank{D},One})::Manifold{D} where {D}
    @assert 0 <= D
    N = D + 1

    if D == 0
        return Manifold(name, OpDict{Int,One}(0 => simplices),
                        OpDict{Int,Int8}(), OpDict{Tuple{Int,Int},One}())
    end

    nvertices, nsimplices = size(simplices)

    # Calculate lower-dimensional simplices
    # See arXiv:1103.3076v2 [cs.NA], section 7
    facelist = Face{N - 1}[]
    for j in 1:size(simplices, 2)
        ks0 = collect(sparse_column_rows(simplices, j))
        @assert length(ks0) == N
        ks = SVector{N,Int}(ks0)
        for n in 1:N
            # Leave out vertex n
            ls = deleteat(ks, n)
            p = bitsign(isodd(n - 1))
            push!(facelist, Face{N - 1}(ls, j, p))
        end
    end
    sort!(facelist; by = f -> f.vertices)

    # Convert facelist into sparse matrix and create boundary operator
    fI = Int[]
    fJ = Int[]
    fV = One[]
    bI = Int[]
    bJ = Int[]
    bV = Int8[]

    nfaces = 0
    oldvertices = SVector{N - 1,Int}(0 for n in 1:(N - 1))
    for f in facelist
        if f.vertices != oldvertices
            oldvertices = f.vertices
            # We found a new face
            nfaces += 1
            for i in f.vertices
                push!(fI, i)
                push!(fJ, nfaces)
                push!(fV, One())
            end
        end
        # Add boundary face
        push!(bI, nfaces)
        push!(bJ, f.parent)
        push!(bV, f.parity)
    end

    faces = SparseOp{Rank{0},Rank{D - 1}}(sparse(fI, fJ, fV, nvertices, nfaces))
    boundaries = SparseOp{Rank{D - 1},Rank{D}}(sparse(bI, bJ, bV, nfaces,
                                                      nsimplices))

    # Recursively create lower-dimensional (D-1)-mfdfold
    mfd1 = Manifold{D - 1}(name, faces)

    # Extend lookup table
    if D > 1
        lookup = map(x -> One(x != 0), boundaries)
        mfd1.lookup[(D - 1, D)] = lookup
        for Ri in 1:(D - 2)
            mfd1.lookup[(Ri, D)] = map(x -> One(x != 0),
                                       mfd1.lookup[(Ri, D - 1)] *
                                       mfd1.lookup[(D - 1, D)])
        end
    end
    for Ri in 1:D, Rj in (Ri + 1):D
        @assert haskey(mfd1.lookup, (Ri, Rj))
    end

    # Create D-manifold
    mfd1.simplices[D] = simplices
    mfd1.boundaries[D] = boundaries
    return Manifold{D}(name, mfd1.simplices, mfd1.boundaries, mfd1.lookup)
end

function Manifold(name::String,
                  simplices::SparseOp{Rank{0},Rank{D},One})::Manifold{D} where {D}
    return Manifold{D}(name, simplices)
end

end
