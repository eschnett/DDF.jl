module Manifolds

using DifferentialForms: bitsign
using SparseArrays
using StaticArrays

using ..Defs
using ..SparseOps
using ..ZeroOrOne

################################################################################

export PrimalDual, Pr, Dl
@enum PrimalDual::Bool Pr Dl

Base.:!(P::PrimalDual) = PrimalDual(!(Bool(P)))

################################################################################

const OpDict{K,T} = Dict{K,SparseOp{<:Any,<:Any,T}} where {K,T}

export Manifold
"""
A discrete manifold 
"""
struct Manifold{D,S}
    name::String

    # If `simplices[R][i,j]` is present, then vertex `i` is part of
    # the `R`-simplex `j`. `R ∈ 0:D`. We could omit `R=0`.
    simplices::OpDict{Int,One}
    # simplices::Dict{Int,Array{Int,2}}

    # The boundary ∂ of `0`-forms vanishes and is not stored. If
    # `boundaries[R][i,j] = s`, then `R`-simplex `j` has
    # `(R-1)`-simplex `i` as boundary with orientation `s`. `R ∈ 1:D`.
    boundaries::OpDict{Int,Int8}

    # Lookup tables from `Ri`-simplices to `Rj`-simplices for `Ri<Rj`.
    # If `lookup[(Ri,Rj)][i,j]` is present, then `Rj`-simplex `j`
    # contains `Ri`-simplex `i`. `Ri ∈ 1:D, Rj ∈ Ri+1:D`. We could
    # omit `Rj=Ri+1`.
    lookup::OpDict{Tuple{Int,Int},One}
    # lookup::Dict{Tuple{Int,Int},Array{Int,2}}

    coords::Array{S,2}

    function Manifold{D,S}(name::String, simplices::OpDict{Int,One},
                           boundaries::OpDict{Int,Int8},
                           lookup::OpDict{Tuple{Int,Int},One},
                           coords::Array{S,2}) where {D,S}
        D::Int
        @assert D >= 0
        @assert Set(keys(simplices)) == Set(0:D)
        @assert Set(keys(boundaries)) == Set(1:D)
        @assert Set(keys(lookup)) ==
                Set((Ri, Rj) for Ri in 1:D for Rj in (Ri + 1):D)
        @assert size(coords, 1) == size(simplices[0], 2)
        @assert size(coords, 2) >= D
        mfd = new{D,S}(name, simplices, boundaries, lookup, coords)
        @assert invariant(mfd)
        return mfd
    end
    function Manifold(name::String, simplices::OpDict{Int,One},
                      boundaries::OpDict{Int,Int8},
                      lookup::OpDict{Tuple{Int,Int},One},
                      coords::Array{S,2}) where {S}
        D = maximum(keys(simplices))
        return Manifold{D,S}(name, simplices, boundaries, lookup, coords)
    end
end
# TODO: Implement also a "cube complex" representation

################################################################################

function Base.show(io::IO, mfd::Manifold{D}) where {D}
    println(io)
    println(io, "Manifold{$D}(")
    println(io, "    name=$(mfd.name)")
    for R in 0:D
        println(io, "    simplices[$R]=$(mfd.simplices[R])")
    end
    for R in 1:D
        print(io, "    boundaries[$R]=$(mfd.boundaries[R])")
    end
    print(io, "    coords=$(mfd.coords)")
    return print(io, ")")
end

function Defs.invariant(mfd::Manifold{D})::Bool where {D}
    D >= 0 || (@assert false; return false)

    # Check simplices
    Set(keys(mfd.simplices)) == Set(0:D) || (@assert false; return false)
    for R in 0:D
        simplices = mfd.simplices[R]::SparseOp{0,R,One}
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
        boundaries = mfd.boundaries[R]::SparseOp{R - 1,R,Int8}
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
        lookup = mfd.lookup[(Ri, Rj)]::SparseOp{Ri,Rj,One}
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
                for k in si     # vertices
                    k ∈ vj || (@assert false; return false)
                end
            end
        end
    end

    size(mfd.coords, 1) == nsimplices(mfd, 0) || (@assert false; return false)
    size(mfd.coords, 2) >= D || (@assert false; return false)

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

function Manifold(name::String, simplices::SparseOp{0,D,One},
                  coords::Array{S,2}) where {D,S}
    @assert 0 <= D
    N = D + 1

    if D == 0
        return Manifold(name, OpDict{Int,One}(0 => simplices),
                        OpDict{Int,Int8}(), OpDict{Tuple{Int,Int},One}(),
                        coords)
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

    faces = SparseOp{0,D - 1}(sparse(fI, fJ, fV, nvertices, nfaces))
    boundaries = SparseOp{D - 1,D}(sparse(bI, bJ, bV, nfaces, nsimplices))

    # Recursively create lower-dimensional (D-1)-manifold
    mfd1 = Manifold(name, faces, coords)

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
    return Manifold(name, mfd1.simplices, mfd1.boundaries, mfd1.lookup, coords)
end

end
