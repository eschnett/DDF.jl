module Manifolds

using DifferentialForms
using SparseArrays
using StaticArrays

using ..Algorithms
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
    # contains `Ri`-simplex `i`. `Ri ∈ 0:D, Rj ∈ Ri+1:D`. We could
    # omit `Rj=Ri+1`.
    lookup::OpDict{Tuple{Int,Int},One}
    # lookup::Dict{Tuple{Int,Int},Array{Int,2}}

    coords::Array{S,2}
    # coords::Vector{SVector{C,S}}
    volumes::Dict{Int,Vector{S}}
    # Coordinates of vertices of dual grid, i.e. circumcentres of
    # primal top-simplices
    dualcoords::Array{S,2}
    # dualcoords::Vector{SVector{C,S}}
    # `dualvolumes[R]` are the volumes of the simplices dual to the
    # primal `R`-simplices
    dualvolumes::Dict{Int,Vector{S}}

    function Manifold{D,S}(name::String, simplices::OpDict{Int,One},
                           boundaries::OpDict{Int,Int8},
                           lookup::OpDict{Tuple{Int,Int},One},
                           coords::Array{S,2}, volumes::Dict{Int,Vector{S}},
                           dualcoords::Array{S,2},
                           dualvolumes::Dict{Int,Vector{S}}) where {D,S}
        D::Int
        @assert D >= 0
        @assert Set(keys(simplices)) == Set(0:D)
        @assert Set(keys(boundaries)) == Set(1:D)
        @assert Set(keys(lookup)) ==
                Set((Ri, Rj) for Ri in 0:D for Rj in (Ri + 1):D)
        @assert size(coords, 1) == size(simplices[0], 2)
        @assert size(coords, 2) >= D
        @assert Set(keys(volumes)) == Set(0:D)
        for R in 0:D
            @assert length(volumes[R]) == size(simplices[R], 2)
        end
        @assert size(dualcoords, 1) == size(simplices[D], 2)
        @assert size(dualcoords, 2) >= D
        @assert Set(keys(dualvolumes)) == Set(0:D)
        for R in 0:D
            @assert length(dualvolumes[R]) == size(simplices[R], 2)
        end
        mfd = new{D,S}(name, simplices, boundaries, lookup, coords, volumes,
                       dualcoords, dualvolumes)
        @assert invariant(mfd)
        return mfd
    end
    function Manifold(name::String, simplices::OpDict{Int,One},
                      boundaries::OpDict{Int,Int8},
                      lookup::OpDict{Tuple{Int,Int},One}, coords::Array{S,2},
                      volumes::Dict{Int,Vector{S}}, dualcoords::Array{S,2},
                      dualvolumes::Dict{Int,Vector{S}}) where {S}
        D = maximum(keys(simplices))
        return Manifold{D,S}(name, simplices, boundaries, lookup, coords,
                             volumes, dualcoords, dualvolumes)
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
        println(io, "    boundaries[$R]=$(mfd.boundaries[R])")
    end
    println(io, "    coords=$(mfd.coords)")
    for R in 0:D
        println(io, "    volumes[$R]=$(mfd.volumes[R])")
    end
    println(io, "    dualcoords=$(mfd.dualcoords)")
    for R in 0:D
        println(io, "    dualvolumes[$R]=$(mfd.dualvolumes[R])")
    end
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
    Set(keys(mfd.lookup)) == Set((Ri, Rj) for Ri in 0:D for Rj in (Ri + 1):D) ||
        (@assert false; return false)
    for Ri in 0:D, Rj in (Ri + 1):D
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

    Set(keys(mfd.volumes)) == Set(0:D) || (@assert false; return false)
    for R in 0:D
        length(mfd.volumes[R]) == nsimplices(mfd, R) ||
            (@assert false; return false)
    end

    size(mfd.dualcoords, 1) == nsimplices(mfd, D) ||
        (@assert false; return false)
    size(mfd.dualcoords, 2) >= D || (@assert false; return false)

    Set(keys(mfd.dualvolumes)) == Set(0:D) || (@assert false; return false)
    for R in 0:D
        length(mfd.dualvolumes[R]) == nsimplices(mfd, R) ||
            (@assert false; return false)
    end

    return true
end

# Comparison

function Base.:(==)(mfd1::Manifold{D}, mfd2::Manifold{D})::Bool where {D}
    return mfd1.simplices[D] == mfd2.simplices[D] && mfd1.coords == mfd2.coords
end
function Base.isequal(mfd1::Manifold{D}, mfd2::Manifold{D})::Bool where {D}
    return isequal(mfd1.simplices[D], mfd2.simplices[D]) &&
           isequal(mfd1.coords, mfd2.coords)
end
function Base.hash(mfd::Manifold{D}, h::UInt) where {D}
    return hash(0x4d34f4ae,
                hash(D, hash(mfd.simplices[D], hash(mfd.coords.op, h))))
end

Base.ndims(::Manifold{D}) where {D} = D

export nsimplices
nsimplices(mfd::Manifold, ::Val{R}) where {R} = size(mfd.simplices[R], 2)
nsimplices(mfd::Manifold, R::Integer) = size(mfd.simplices[R], 2)

################################################################################

# Outer constructor

export ZeroVolumeException
struct ZeroVolumeException <: Exception
    D::Int
    i::Int
    simplex::Vector{Int}
    cs::Vector                  # Vector{SVector{D,T}}
end

struct Face{N}
    vertices::SVector{N,Int}
    parent::Int
    parity::Int8
end

function Manifold(name::String, simplices::SparseOp{0,D,One},
                  coords::Array{S,2}) where {D,S}
    @assert 0 <= D
    N = D + 1

    nvertices, nsimplices = size(simplices)
    @assert size(coords, 1) == nvertices
    @assert size(coords, 2) >= D

    if D == 0
        volumes = fill(S(1), nsimplices)
        dualcoords = coords
        dualvolumes = fill(S(1), nvertices)
        return Manifold(name, OpDict{Int,One}(0 => simplices),
                        OpDict{Int,Int8}(), OpDict{Tuple{Int,Int},One}(),
                        coords, Dict{Int,Vector{S}}(0 => volumes), dualcoords,
                        Dict{Int,Vector{S}}(0 => dualvolumes))
    end

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
    oldvertices = zero(SVector{N - 1,Int})
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
    mfd1.lookup[(0, D)] = simplices
    if D > 1
        lookup = map(x -> One(x != 0), boundaries)
        mfd1.lookup[(D - 1, D)] = lookup
        for Ri in 1:(D - 2)
            mfd1.lookup[(Ri, D)] = map(x -> One(x != 0),
                                       mfd1.lookup[(Ri, D - 1)] *
                                       mfd1.lookup[(D - 1, D)])
        end
    end
    for Ri in 0:D, Rj in (Ri + 1):D
        @assert haskey(mfd1.lookup, (Ri, Rj))
    end

    # Calculate volumes, dual coordinates, and dual volumes
    C = size(coords, 2)
    volumes = calc_volumes(Val(D), Val(C), simplices, coords)

    dualcoords = calc_dualcoords(Val(D), Val(C), simplices, coords)

    # TODO: re-use dualvolumes from mfd1 (is this possible?)
    dualvolumes = Dict{Int,Vector{S}}()
    if C == D
        # Only set correctly for the final manifold
        dualvolumes[D] = fill(S(1), nsimplices)
        for R in (D - 1):-1:0
            dualvolumes[R] = calc_dualvolumes(Val(D), Val(R), Val(C),
                                              mfd1.simplices[R],
                                              R + 1 == D ? simplices :
                                              mfd1.simplices[R + 1],
                                              mfd1.lookup[(R, R + 1)]', coords,
                                              dualvolumes[R + 1])
        end
    else
        # dummy data, won't be used
        dualvolumes[D] = zeros(S, nsimplices)
        for R in 0:(D - 1)
            dualvolumes[R] = zeros(S, size(mfd1.simplices[R], 2))
        end
    end

    # Only test for the final manifold
    if C == D
        check_delaunay(Val(D), Val(C), simplices, mfd1.lookup[(D - 1, D)],
                       mfd1.lookup[(D - 1, D)]', coords, dualcoords)
    end

    # Create D-manifold
    mfd1.simplices[D] = simplices
    mfd1.boundaries[D] = boundaries
    mfd1.volumes[D] = volumes
    # ignoring `mfd1` dual coordinates
    # ignoring 'mfd1` dual volumes
    return Manifold(name, mfd1.simplices, mfd1.boundaries, mfd1.lookup, coords,
                    mfd1.volumes, dualcoords, dualvolumes)
end

################################################################################

"""
Calculate volumes
"""
function calc_volumes(::Val{D}, ::Val{C}, simplices::SparseOp{0,D,One},
                      coords::Array{S,2}) where {D,C,S}
    nvertices, nsimplices = size(simplices)
    volumes = Array{S}(undef, nsimplices)
    for i in 1:nsimplices
        si = sparse_column_rows(simplices, i)
        @assert length(si) == D + 1
        xs = SVector{D + 1}(Form{C,1}(SVector{C,S}(@view coords[i, :]))
                            for i in si)
        volumes[i] = volume(xs)
    end
    return volumes
end

"""
Calculate dual coordinates, i.e. circumcentres
"""
function calc_dualcoords(::Val{D}, ::Val{C}, simplices::SparseOp{0,D,One},
                         coords::Array{S,2}) where {D,C,S}
    nvertices, nsimplices = size(simplices)
    dualcoords = Array{S}(undef, nsimplices, C)
    for i in 1:nsimplices
        si = sparse_column_rows(simplices, i)
        @assert length(si) == D + 1
        xs = SVector{D + 1}(Form{C,1}(SVector{C,S}(@view coords[i, :]))
                            for i in si)
        dualcoords[i, :] .= circumcentre(xs)
    end
    return dualcoords
end

"""
Calculate circumcentric dual volumes

See [1198555.1198667, section 6.2.1]
"""
function calc_dualvolumes(::Val{D}, ::Val{R}, ::Val{C},
                          simplices::SparseOp{0,R,One},
                          simplices1::SparseOp{0,R1,One},
                          parents::SparseOp{R1,R,One}, coords::Array{S,2},
                          dualvolumes1::Vector{S}) where {D,R,R1,C,S}
    D::Int
    R::Int
    R1::Int
    C::Int
    @assert 0 <= R <= R1 <= D
    @assert R1 == R + 1
    @assert size(coords, 2) == C
    nvertices, nsimplices = size(simplices)
    dualvolumes = Array{S}(undef, nsimplices)
    # Loop over all `R`-simplices
    for i in 1:nsimplices
        si = sparse_column_rows(simplices, i)
        @assert length(si) == R + 1
        xsi = SVector{R + 1}(Form{C,1}(SVector{C,S}(@view coords[i, :]))
                             for i in si)
        # bci = sum(xsi) / length(xsi)
        # TODO: this could be dualcoords
        cci = circumcentre(xsi)
        vol = zero(S)
        # Loop over all neighbouring `R+1`-simplices
        for j in sparse_column_rows(parents, i)
            sj = sparse_column_rows(simplices1, j)
            @assert length(sj) == R + 2

            b = dualvolumes1[j]
            # TODO: Calculate lower-rank circumcentres as intersection
            # between boundary and the line connecting two simplices?
            # TODO: Cache circumcentres ahead of time
            # TODO: could be dualcoords
            xsj = SVector{R + 2}(Form{C,1}(SVector{C,S}(@view coords[j, :]))
                                 for j in sj)
            bcj = sum(xsj) / length(xsj)
            ccj = circumcentre(xsj)
            # Handle case where the volume should be negative, i.e.
            # when the volume circumcentre cci is on the "other" side
            # of the face circumcentre ccj. See [arXiv:1204.0747
            # [cs.CG]].
            ysi = map(x -> x - xsi[1], deleteat(xsi, 1))
            v1 = ∧(ysi..., bcj - xsi[1])
            v2 = ∧(ysi..., ccj - xsi[1])
            s = v1 ⋅ v2
            s = sign(s[])
            # # TODO
            # if !(s == 1)
            #     @show D R simplices coords i j xsi cci xsj ccj ysi v1 v2 s
            # end
            # @assert s == 1
            h = norm(cci - ccj)
            vol += b * s * h
        end
        # We want well-centred meshes
        # @assert vol > 0
        dualvolumes[i] = vol / (D - R)
    end
    return dualvolumes
end

"""
Check Delaunay condition: No vertex must lie in the circumcentre of a
simplex
"""
function check_delaunay(::Val{D}, ::Val{C}, simplices::SparseOp{0,D,One},
                        lookup::SparseOp{D1,D,One}, lookup1::SparseOp{D,D1,One},
                        coords::Array{S,2},
                        dualcoords::Array{S,2}) where {D,D1,C,S}
    D::Int
    D1::Int
    @assert 0 <= D1 <= D
    @assert D1 == D - 1
    C::Int
    for i in 1:size(simplices, 2)
        si = sparse_column_rows(simplices, i)
        @assert length(si) == D + 1
        x1i = Form{C,1}(SVector{C,S}(@view coords[first(si), :]))
        cci = Form{C,1}(SVector{C,S}(@view dualcoords[i, :]))
        cri2 = norm2(x1i - cci)
        # Loop over all faces
        for j in sparse_column_rows(lookup, i)
            # Loop over all simplices (except i)
            for k in sparse_column_rows(lookup1, j)
                if k != i
                    # Loop over all vertices
                    for l in sparse_column_rows(simplices, k)
                        # Ignore vertices of simplex i
                        if l ∉ si
                            xl = Form{C,1}(SVector{C,S}(@view coords[l, :]))
                            d2 = norm2(xl - cci)
                            @assert d2 >= cri2 || d2 ≈ cri2
                        end
                    end
                end
            end
        end
    end
end

end
