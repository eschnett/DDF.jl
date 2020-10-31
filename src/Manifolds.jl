module Manifolds

using ComputedFieldTypes
using DifferentialForms
using LinearAlgebra
using NearestNeighbors
using SparseArrays
using StaticArrays

using ..Algorithms
using ..Defs
using ..Meshing
using ..SparseOps
using ..ZeroOrOne

################################################################################

export PrimalDual, Pr, Dl
@enum PrimalDual::Bool Pr Dl

Base.:!(P::PrimalDual) = PrimalDual(!(Bool(P)))

################################################################################

export DualKind, BarycentricDuals, CircumcentricDuals
@enum DualKind BarycentricDuals CircumcentricDuals

################################################################################

const Maybe{T} = Union{Nothing,T}

const OpDict{K,T} = Dict{K,SparseOp{<:Any,<:Any,T}} where {K,T}

export Manifold
"""
A discrete manifold 
"""
mutable struct Manifold{D,C,S}
    name::String

    dualkind::DualKind
    use_weighted_duals::Bool

    # If `simplices[R][i,j]` is present, then vertex `i` is part of
    # the `R`-simplex `j`. `R ∈ 0:D`. We could omit `R=0`.
    simplices::OpDict{Int,One}
    # simplices::Dict{Int,Array{Int,2}}

    # The boundary ∂ of `0`-forms vanishes and is not stored. If
    # `boundaries[R][i,j] = s`, then `R`-simplex `j` has
    # `(R-1)`-simplex `i` as boundary with orientation `s`. `R ∈ 1:D`.
    boundaries::OpDict{Int,Int8}
    _isboundary::OpDict{Int,One}

    # Lookup tables from `Ri`-simplices to `Rj`-simplices. If
    # `lookup[(Ri,Rj)][i,j]` is present, then `Rj`-simplex `j`
    # contains `Ri`-simplex `i`. `Ri ∈ 0:D, Rj ∈ 0:D`. Many of these
    # are trivial and could be omitted.
    _lookup::OpDict{Tuple{Int,Int},One}
    # lookup::Dict{Tuple{Int,Int},Array{Int,2}}

    # coords::Array{S,2}
    coords::Dict{Int,Vector{SVector{C,S}}}
    volumes::Dict{Int,Vector{S}}

    # Vertex weights [arXiv:math/0508188], [DOI:10.1145/2602143]
    weights::Vector{S}

    # Coordinates of vertices of dual grid, i.e.
    # barycentres/circumcentres of primal top-simplices
    # dualcoords::Array{S,2}
    _dualcoords::Dict{Int,Vector{SVector{C,S}}}
    # `dualvolumes[R]` are the volumes of the simplices dual to the
    # primal `R`-simplices
    _dualvolumes::Dict{Int,Vector{S}}

    # Nearest neighbour tree for simplex vertices
    _simplex_tree::Maybe{KDTree{SVector{C,S},Euclidean,S}}

    function Manifold{D,C,S}(name::String, dualkind::DualKind,
                             use_weighted_duals::Bool,
                             simplices::OpDict{Int,One},
                             boundaries::OpDict{Int,Int8},
                             coords::Dict{Int,Vector{SVector{C,S}}},
                             volumes::Dict{Int,Vector{S}},
                             weights::Vector{S}) where {D,C,S}
        D::Int
        @assert 0 ≤ D ≤ C
        mfd = new{D,C,S}(name, dualkind, use_weighted_duals, simplices,
                         boundaries, OpDict{Int,One}(),
                         OpDict{Tuple{Int,Int},One}(), coords, volumes, weights,
                         Dict{Int,Vector{SVector{C,S}}}(),
                         Dict{Int,AbstractVector{S}}(), nothing)
        @assert invariant(mfd)
        return mfd
    end
    function Manifold(name::String, dualkind::DualKind,
                      use_weighted_duals::Bool, simplices::OpDict{Int,One},
                      boundaries::OpDict{Int,Int8},
                      coords::Dict{Int,Vector{SVector{C,S}}},
                      volumes::Dict{Int,Vector{S}},
                      weights::Vector{S}) where {C,S}
        D = maximum(keys(simplices))
        return Manifold{D,C,S}(name, dualkind, use_weighted_duals, simplices,
                               boundaries, coords, volumes, weights)
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
    for R in 0:(D - 1)
        if haskey(mfd.isboundary, R)
            println(io, "    isboundary[$R]=$(mfd.isboundary[R])")
        end
    end
    for R in 0:D
        println(io, "    coords[$R]=$(mfd.coords[R])")
    end
    for R in 0:D
        println(io, "    volumes[$R]=$(mfd.volumes[R])")
    end
    println(io, "    weights=$(mfd.weights)")
    for R in 0:D
        if haskey(mfd.dualcoords, R)
            println(io, "    dualcoords[$R]=$(mfd.dualcoords[R])")
        end
    end
    for R in 0:D
        if haskey(mfd.dualvolumes, R)
            println(io, "    dualvolumes[$R]=$(mfd.dualvolumes[R])")
        end
    end
    return print(io, ")")
end

function Defs.invariant(mfd::Manifold{D,C})::Bool where {D,C}
    D ≥ 0 || (@assert false; return false)
    C ≥ D || (@assert false; return false)

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

    # # Check isboundary
    # Set(keys(mfd.isboundary)) == Set(0:(D - 1)) || (@assert false; return false)
    # for R in 0:(D - 1)
    #     isboundary = mfd.isboundary[R]::SparseOp{R,R,One}
    #     size(isboundary) == (nsimplices(mfd, R), nsimplices(mfd, R)) ||
    #         (@assert false; return false)
    # end
    # # TODO: check content as well

    # # Check lookup tables
    # Set(keys(mfd.lookup)) == Set((Ri, Rj) for Ri in 0:D for Rj in 0:D) ||
    #     (@assert false; return false)
    # for Ri in 0:D, Rj in 0:D
    #     lookup = mfd.lookup[(Ri, Rj)]::SparseOp{Ri,Rj,One}
    #     size(lookup) == (nsimplices(mfd, Ri), nsimplices(mfd, Rj)) ||
    #         (@assert false; return false)
    #     if Rj ≥ Ri
    #         for j in 1:size(lookup, 2) # Rj-simplex
    #             vj = sparse_column_rows(mfd.simplices[Rj], j)
    #             length(vj) == Rj + 1 || (@assert false; return false)
    #             sj = sparse_column_rows(lookup, j)
    #             length(sj) == binomial(Rj + 1, Ri + 1) ||
    #                 (@assert false; return false)
    #             for i in sj         # Ri-simplex
    #                 si = sparse_column_rows(mfd.simplices[Ri], i)
    #                 for k in si     # vertices
    #                     k ∈ vj || (@assert false; return false)
    #                 end
    #             end
    #         end
    #         mfd.lookup[(Ri, Rj)] == mfd.lookup[(Rj, Ri)]' ||
    #             (@assert false; return false)
    #     end
    # end

    Set(keys(mfd.coords)) == Set(0:D) || (@assert false; return false)
    for R in 0:D
        length(mfd.coords[R]) == nsimplices(mfd, R) ||
            (@assert false; return false)
    end

    Set(keys(mfd.volumes)) == Set(0:D) || (@assert false; return false)
    for R in 0:D
        length(mfd.volumes[R]) == nsimplices(mfd, R) ||
            (@assert false; return false)
    end

    length(mfd.weights) == nsimplices(mfd, 0) || (@assert false; return false)

    # Set(keys(mfd.dualcoords)) == Set(0:D) || (@assert false; return false)
    # for R in 0:D
    #     length(mfd.dualcoords[R]) == nsimplices(mfd, R) ||
    #         (@assert false; return false)
    # end

    # Set(keys(mfd.dualvolumes)) == Set(0:D) || (@assert false; return false)
    # for R in 0:D
    #     length(mfd.dualvolumes[R]) == nsimplices(mfd, R) ||
    #         (@assert false; return false)
    # end

    return true
end

# Comparison

function Base.:(==)(mfd1::Manifold{D,C}, mfd2::Manifold{D,C})::Bool where {D,C}
    mfd1 ≡ mfd2 && return true
    return mfd1.simplices[D] == mfd2.simplices[D] &&
           mfd1.coords[0] == mfd2.coords[0] &&
           mfd1.weights == mfd2.weights
end
function Base.isequal(mfd1::Manifold{D,C},
                      mfd2::Manifold{D,C})::Bool where {D,C}
    return isequal(mfd1.simplices[D], mfd2.simplices[D]) &&
           isequal(mfd1.coords[0], mfd2.coords[0]) &&
           isequal(mfd1.weights, mfd2.weights)
end
function Base.hash(mfd::Manifold{D,C}, h::UInt) where {D,C}
    return hash(0x4d34f4ae,
                hash(D,
                     hash(C,
                          hash(mfd.simplices[D],
                               hash(mfd.coords[0], hash(mfd.weights[0], h))))))
end

Base.ndims(::Manifold{D}) where {D} = D

export nsimplices
nsimplices(mfd::Manifold, ::Val{R}) where {R} = size(mfd.simplices[R], 2)
nsimplices(mfd::Manifold, R::Integer) = size(mfd.simplices[R], 2)

export random_point
"""
Return random point ∈ manifold
"""
function random_point(::Val{R}, mfd::Manifold{D,C,S}) where {D,C,R,S}
    @assert 0 ≤ R ≤ D ≤ C
    C == 0 && return SVector{C,S}()
    N = R + 1
    # Choose simplex
    i = rand(1:nsimplices(mfd, R))
    si = sparse_column_rows(mfd.simplices[R], i)
    @assert length(si) == R + 1
    # Choose point ∈ simplex
    λ = abs.(randn(SVector{C + 1,S}))
    λ /= sum(λ)
    x = sum(λ[n] * mfd.coords[0][si[n]] for n in 1:N)
    return x::SVector{C,S}
end

################################################################################

# Outer constructor

function Manifold(name::String, simplicesD::SparseOp{0,D,One},
                  coords0::Vector{SVector{C,S}}, weights::Vector{S};
                  dualkind::DualKind=CircumcentricDuals,
                  optimize_mesh::Bool=true,
                  use_weighted_duals::Bool=true) where {D,C,S}
    @assert 0 ≤ D ≤ C

    nvertices, nsimplices = size(simplicesD)
    @assert length(coords0) == nvertices

    # Calculate lower-Dimensional simplices
    simplices = OpDict{Int,One}(D => simplicesD)
    boundaries = OpDict{Int,Int8}()
    for R in D:-1:1
        facesR, boundariesR = calc_faces_boundaries(simplices[R])
        simplices[R - 1] = facesR
        boundaries[R] = boundariesR
    end

    # Calculate coordinates and volumes
    coords = Dict{Int,Vector{SVector{C,S}}}()
    volumes = Dict{Int,Vector{S}}()
    for R in 0:D
        # TODO: Combine these two calculations
        coords[R] = calc_coords(simplices[R], coords0)
        volumes[R] = calc_volumes(simplices[R], coords0)
        if !(all(x -> x != 0 && isfinite(x), volumes[R]))
            @show volumes[R]
            @show D R
        end
        @assert all(x -> x != 0 && isfinite(x), volumes[R])
    end

    if optimize_mesh && use_weighted_duals
        @assert dualkind == CircumcentricDuals
        # Optimize weights
        # Calculate mask for boundary vertices (which must not be moved)
        if D > 0
            boundary_faces = zeros(Int8, size(boundaries[D], 1))
            for j in 1:size(boundaries[D], 2)
                for (i, s) in sparse_column(boundaries[D], j)
                    boundary_faces[i] += s
                end
            end
            # This might indicate a severe bug; shouldn't faces have
            # opposite orientations when viewed from two neighbouring
            # simplices?
            # @assert all(s -> -1 ≤ s ≤ 1, boundary_faces)
            @assert all(s -> -2 ≤ s ≤ 2, boundary_faces)

            # @assert C == D
            coords0 = coords[0]
            dof = ones(SMatrix{C,C,S}, length(coords0))
            @assert length(boundary_faces) == size(simplices[D - 1], 2)
            for j in 1:size(simplices[D - 1], 2)
                if isodd(boundary_faces[j])
                    sj = sparse_column_rows(simplices[D - 1], j)
                    sj = SVector{D,Int}(sj[n] for n in 1:D)
                    # initially, no directions are allowed
                    P = zero(SMatrix{C,C,S})
                    for n in 2:D
                        x = coords0[sj[n]] - coords0[sj[1]]
                        # ensure x is orthogonal to the directions in P
                        x -= P * x
                        Q = x * x' / norm2(x)
                        # add one allowed direction
                        P += Q
                    end
                    for i in sparse_column_rows(simplices[D - 1], j)
                        dof[i] = P * dof[i] * P
                        dof[i] = (dof[i] + dof[i]') / 2
                    end
                end
            end

        else
            dof = zeros(SMatrix{C,C,S}, length(coords[0]))
        end
        dof::Vector{SMatrix{C,C,S}}
        coords0, weights = optimize_mesh1(Val(dualkind), Val(D), simplices,
                                          coords[0], dof, weights)
        # Re-calculate coordinates and volumes
        coords = Dict{Int,Vector{SVector{C,S}}}()
        volumes = Dict{Int,Vector{S}}()
        for R in 0:D
            # TODO: Combine these two calculations
            coords[R] = calc_coords(simplices[R], coords0)
            volumes[R] = calc_volumes(simplices[R], coords0)
            @assert all(x -> x != 0 && isfinite(x), volumes[R])
        end
    end

    # TODO: Move this to invariant, maybe add `haskey` for dualcoords
    # if D > 0
    #     check_delaunay(simplices[D], lookup[(D - 1, D)], lookup[(D, D - 1)],
    #                    coords[0], dualcoords[D])
    # end

    # Create D-manifold
    return Manifold(name, dualkind, use_weighted_duals, simplices, boundaries,
                    coords, volumes, weights)
end

################################################################################

export lookup
@inline function lookup(::Val{Ri}, ::Val{Rj},
                        mfd::Manifold{D,C,S}) where {Ri,Rj,D,C,S}
    @assert 0 ≤ Rj ≤ D
    @assert 0 ≤ Ri ≤ D
    !haskey(mfd._lookup, (Ri, Rj)) && calc_lookup!(mfd, Ri, Rj)
    return mfd._lookup[(Ri, Rj)]::SparseOp{Ri,Rj,One}
end
@inline function lookup(Ri::Int, Rj::Int, mfd::Manifold{D,C,S}) where {D,C,S}
    return lookup(Val(Ri), Val(Rj), mfd)
end

function calc_lookup!(mfd::Manifold{D,C,S}, Ri::Int, Rj::Int) where {D,C,S}
    @assert 0 ≤ Rj ≤ D
    @assert 0 ≤ Ri ≤ D
    @assert !haskey(mfd._lookup, (Ri, Rj))

    sizei = size(mfd.simplices[Ri], 2)
    sizej = size(mfd.simplices[Rj], 2)

    if Ri == 0
        # Simplex definitions
        mfd._lookup[(Ri, Rj)] = mfd.simplices[Rj]
    elseif Ri == Rj
        # Identity
        mfd._lookup[(Ri, Rj)] = SparseOp{Ri,Rj}(sparse(1:sizei, 1:sizej,
                                                       fill(One(), sizei)))
    elseif Ri == Rj - 1
        # Absolute value of boundaries
        mfd._lookup[(Ri, Rj)] = map(x -> One(x ≠ 0), mfd.boundaries[Rj])
    elseif Ri < Rj - 1
        # Chain of two lookup tables
        Rk = Rj - 1
        mfd._lookup[(Ri, Rj)] = map(x -> One(x ≠ 0),
                                    lookup(Val(Ri), Val(Rk), mfd) *
                                    lookup(Val(Rk), Val(Rj), mfd))
    else
        # Transpose
        mfd._lookup[(Ri, Rj)] = lookup(Val(Rj), Val(Ri), mfd)'
    end

    return nothing
end

export isboundary
@inline function isboundary(::Val{R}, mfd::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R < D
    !haskey(mfd._isboundary, R) && calc_isboundary!(mfd, R)
    return mfd._isboundary[R]::SparseOp{R,R,One}
end
@inline function isboundary(R::Int, mfd::Manifold{D,C,S}) where {D,C,S}
    return isboundary(Val(R), mfd)
end

function calc_isboundary!(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    @assert 0 ≤ R < D
    @assert !haskey(mfd._isboundary, R)

    I = Int[]
    J = Int[]
    V = One[]
    nelts = nsimplices(mfd, R)
    if R == D - 1
        isbndface = falses(size(mfd.boundaries[D], 1))
        for j in 1:size(mfd.boundaries[D], 2)
            for i in sparse_column_rows(mfd.boundaries[D], j)
                isbndface[i] = !isbndface[i]
            end
        end
        for (i, b) in enumerate(isbndface)
            if b
                push!(I, i)
                push!(J, i)
                push!(V, One())
            end
        end
    else
        lup = lookup(Val(R), Val(D - 1), mfd)::SparseOp{R,D - 1,One}
        for j in findnz(isboundary(D - 1, mfd).op)[1]
            for i in sparse_column_rows(lup, j)
                push!(I, i)
                push!(J, i)
                push!(V, One())
            end
        end
    end
    mfd._isboundary[R] = SparseOp{R,R}(sparse(I, J, V, nelts, nelts, max))

    return nothing
end

export dualcoords
@inline function dualcoords(::Val{R}, mfd::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R ≤ D
    !haskey(mfd._dualcoords, R) && calc_dualcoords!(mfd, R)
    return mfd._dualcoords[R]::Vector{SVector{C,S}}
end
@inline function dualcoords(R::Int, mfd::Manifold{D,C,S}) where {D,C,S}
    return dualcoords(Val(R), mfd)
end

function calc_dualcoords!(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    @assert 0 ≤ R ≤ D
    @assert !haskey(mfd._dualcoords, R)
    if mfd.use_weighted_duals
        mfd._dualcoords[R] = calc_dualcoords(Val(mfd.dualkind),
                                             mfd.simplices[R], mfd.coords[0],
                                             mfd.weights)
    else
        mfd._dualcoords[R] = calc_dualcoords(Val(mfd.dualkind),
                                             mfd.simplices[R], mfd.coords[0])
    end
    return nothing
end

export dualvolumes
@inline function dualvolumes(::Val{R}, mfd::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R ≤ D
    !haskey(mfd._dualvolumes, R) && calc_dualvolumes!(mfd, R)
    return mfd._dualvolumes[R]::Vector{S}
end
@inline function dualvolumes(R::Int, mfd::Manifold{D,C,S}) where {D,C,S}
    return dualvolumes(Val(R), mfd)
end

function calc_dualvolumes!(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    @assert 0 ≤ R ≤ D
    @assert !haskey(mfd._dualvolumes, R)
    if mfd.dualkind == BarycentricDuals
        mfd._dualvolumes[R] = calc_dualvolumes(Val(mfd.dualkind), Val(D),
                                               Val(R), mfd.simplices[R],
                                               mfd.lookup, mfd.coords[0],
                                               mfd.volumes[D])
        @assert all(x -> x != 0 && isfinite(x), mfd.dualvolumes[R])
    elseif mfd.dualkind == CircumcentricDuals
        if R == D
            mfd._dualvolumes[D] = fill(S(1), nsimplices(mfd, D))
        else
            mfd._dualvolumes[R] = calc_dualvolumes(Val(mfd.dualkind), Val(D),
                                                   mfd.simplices[R],
                                                   mfd.simplices[R + 1],
                                                   lookup(Val(R + 1), Val(R),
                                                          mfd), mfd.coords[0],
                                                   mfd.volumes[R],
                                                   dualcoords(R, mfd),
                                                   dualcoords(R + 1, mfd),
                                                   dualvolumes(R + 1, mfd))
        end
    else
        @assert false
    end

    # Ensure that all dual volumes are strictly positive
    allpositive = all(>(0), mfd._dualvolumes[R])
    if !allpositive
        @warn "Not all dual $R-volumes are strictly positive"
        @show count(≤(0), mfd._dualvolumes[R])
        @show findmin(mfd._dualvolumes[R])
        @show dualcoords(R, mfd)[findmin(mfd._dualvolumes[R])[2]]
    end

    return nothing
end

export simplex_tree
@inline function simplex_tree(mfd::Manifold{D,C,S}) where {D,C,S}
    mfd._simplex_tree === nothing && calc_simplex_tree!(mfd)
    return mfd._simplex_tree::KDTree{SVector{C,S},Euclidean,S}
end

function calc_simplex_tree!(mfd::Manifold{D,C,S}) where {D,C,S}
    @assert mfd._simplex_tree === nothing
    mfd._simplex_tree = KDTree(mfd.coords[0])
    return nothing
end

################################################################################

struct Face{N}
    vertices::SVector{N,Int}
    parent::Int
    parity::Int8
end

"""
Calculate faces and boundaries
"""
function calc_faces_boundaries(simplices::SparseOp{0,D,One}) where {D}
    D::Int
    @assert 0 < D

    nvertices, nsimplices = size(simplices)

    # See arXiv:1103.3076v2 [cs.NA], section 7
    facelist = Face{D}[]
    for j in 1:size(simplices, 2)
        ks = SVector{D + 1,Int}(sparse_column_rows(simplices, j)[n]
                                for n in 1:(D + 1))
        for n in 1:(D + 1)
            # Leave out vertex n
            ls = deleteat(ks, n)
            p = bitsign(isodd(n - 1))
            push!(facelist, Face{D}(ls, j, p))
        end
    end
    sort!(facelist; by=f -> f.vertices)

    # Convert facelist into sparse matrix and create boundary operator
    fI = Int[]
    fJ = Int[]
    fV = One[]
    bI = Int[]
    bJ = Int[]
    bV = Int8[]

    nfaces = 0
    oldvertices = zero(SVector{D,Int})
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

    return faces, boundaries
end

################################################################################

function optimize_mesh1(::Val{CircumcentricDuals}, ::Val{D},
                        simplices::OpDict{Int,One},
                        coords::Vector{SVector{C,S}},
                        dof::Vector{SMatrix{C,C,S}},
                        weights::Vector{S}) where {D,C,S}
    D::Int
    C::Int
    @assert 0 ≤ D ≤ C

    for D′ in D:D
        α = S(1) / 2D′
        for iter in 1:100
            new_coords = copy(coords)
            new_weights = copy(weights)
            for R in 1:D′
                shift_cs, shift_ws = improve_mesh(simplices[R], coords, weights)
                for i in 1:length(new_coords)
                    # nnzev = count(isapprox(1), eigvals(dof[i]))
                    # if nnzev ≤ D′
                    # new_coords[i] += α * dof[i] * shift_cs[i]
                    new_weights[i] += α * shift_ws[i]
                    # end
                end
            end
            new_weights .-= sum(new_weights) / length(new_weights)
            if !(sum(new_weights) + 1 ≈ 1)
                @show maximum(abs.(new_weights)) sum(new_weights)
            end
            @assert sum(new_weights) + 1 ≈ 1
            coords = new_coords
            weights = new_weights
        end
    end

    return coords, weights
end

################################################################################

"""
Calculate coordinates
"""
function calc_coords(simplices::SparseOp{0,D,One},
                     coords0::Vector{SVector{C,S}}) where {D,C,S}
    nvertices, nsimplices = size(simplices)
    D == 0 && return coords0
    coords = Array{SVector{C,S}}(undef, nsimplices)
    @inbounds for i in 1:nsimplices
        si = sparse_column_rows(simplices, i)
        xs = SVector{D + 1}(coords0[i] for i in si)
        coords[i] = barycentre(xs)
    end
    return coords
end

"""
Calculate volumes
"""
function calc_volumes(simplices::SparseOp{0,D,One},
                      coords::Vector{SVector{C,S}}) where {D,C,S}
    nvertices, nsimplices = size(simplices)
    D == 0 && return fill(one(S), nvertices)
    volumes = Array{S}(undef, nsimplices)
    @inbounds for i in 1:nsimplices
        si = sparse_column_rows(simplices, i)
        xs = SVector{D + 1}(Form{C,1}(coords[i]) for i in si)
        volumes[i] = volume(xs)
    end
    return volumes
end

"""
Calculate barycentric dual coordinates
"""
function calc_dualcoords(::Val{BarycentricDuals}, simplices::SparseOp{0,D,One},
                         coords::Vector{SVector{C,S}}) where {D,C,S}
    D == 0 && return coords
    nsimplices = size(simplices, 2)
    dualcoords = Array{SVector{C,S}}(undef, nsimplices)
    @inbounds for i in 1:nsimplices
        si = sparse_column_rows(simplices, i)
        xsi = SVector{D + 1}(Form{C,1}(coords[i]) for i in si)
        dualcoords[i] = barycentre(xsi)
    end
    return dualcoords
end

"""
Calculate circumcentric dual coordinates
"""
function calc_dualcoords(::Val{CircumcentricDuals},
                         simplices::SparseOp{0,D,One},
                         coords::Vector{SVector{C,S}}) where {D,C,S}
    nvertices, nsimplices = size(simplices)
    @assert length(coords) == nvertices
    D == 0 && return coords
    dualcoords = Array{SVector{C,S}}(undef, nsimplices)
    @inbounds for i in 1:nsimplices
        si = SVector{D + 1,Int}(sparse_column_rows(simplices, i)[n]
                                for n in 1:(D + 1))
        xs = SVector{D + 1}(Form{C,1}(coords[i]) for i in si)
        dualcoords[i] = circumcentre(xs)
    end
    return dualcoords
end

"""
Calculate weighted circumcentric dual coordinates
"""
function calc_dualcoords(::Val{CircumcentricDuals},
                         simplices::SparseOp{0,D,One},
                         coords::Vector{SVector{C,S}},
                         weights::AbstractVector{S}) where {D,C,S}
    nvertices, nsimplices = size(simplices)
    @assert length(coords) == nvertices
    @assert length(weights) == nvertices
    D == 0 && return coords
    dualcoords = Array{SVector{C,S}}(undef, nsimplices)
    @inbounds for i in 1:nsimplices
        si = SVector{D + 1,Int}(sparse_column_rows(simplices, i)[n]
                                for n in 1:(D + 1))
        xs = SVector{D + 1}(Form{C,1}(coords[i]) for i in si)
        ws = SVector{D + 1}(Form{C,0}((weights[i],)) for i in si)
        dualcoords[i] = circumcentre(xs, ws)
    end
    return dualcoords
end

"""
Calculate barycentric dual volumes
"""
function calc_dualvolumes(::Val{BarycentricDuals}, ::Val{D}, ::Val{R},
                          simplices::SparseOp{0,R,One},
                          lookup::OpDict{Tuple{Int,Int},One},
                          coords::Vector{SVector{C,S}}, volumes::Vector{S},
                          dualcoords::Dict{Int,Vector{SVector{C,S}}}) where {D,
                                                                             R,
                                                                             C,
                                                                             S}
    D::Int
    R::Int
    C::Int
    @assert 0 ≤ R ≤ D ≤ C

    dualvolumes = Array{S}(undef, size(simplices, 2))

    if R == D
        @inbounds for i in 1:size(simplices, 2)
            dualvolumes[i] = 1
        end

    elseif R == 0
        lookup_D_R = lookup[(D, R)]::SparseOp{D,R,One}

        for i in 1:size(simplices, 2)
            vol = S(0)
            for j in sparse_column_rows(lookup_D_R, i)
                vol += volumes[j]
            end
            dualvolumes[i] = vol / (D + 1)
        end

    elseif R == D - 1
        lookup_D_R = lookup[(D, R)]::SparseOp{D,R,One}
        dualcoords_R = dualcoords[R]
        dualcoords_D = dualcoords[D]

        @inbounds for i in 1:size(simplices, 2)
            bci = dualcoords_R[i]

            vol = S(0)
            for j in sparse_column_rows(lookup_D_R, i)
                bcj = dualcoords_D[j]

                vol += volume(SVector(bci, bcj))
            end
            dualvolumes[i] = vol
        end

    elseif R == D - 2
        lookup_D_R = lookup[(D, R)]::SparseOp{D,R,One}
        lookup_D1_D = lookup[(D - 1, D)]::SparseOp{D - 1,D,One}
        lookup_R_D1 = lookup[(R, D - 1)]::SparseOp{R,D - 1,One}
        dualcoords_R = dualcoords[R]
        dualcoords_D = dualcoords[D]
        dualcoords_D1 = dualcoords[D - 1]

        # Loop over all `R`-forms
        @inbounds for i in 1:size(simplices, 2)
            bci = dualcoords_R[i]

            vol = S(0)
            # Loop over all neighbouring `D`-forms
            for j in sparse_column_rows(lookup_D_R, i)
                bcj = dualcoords_D[j]

                # Loop over all contained `D-1`-forms that contain `i`
                for k in sparse_column_rows(lookup_D1_D, j)
                    if i ∈ sparse_column_rows(lookup_R_D1, k)
                        bck = dualcoords_D1[k]

                        vol += volume(SVector(bci, bcj, bck))
                    end
                end
            end
            dualvolumes[i] = vol
        end

    elseif R == D - 3
        lookup_D_R = lookup[(D, R)]::SparseOp{D,R,One}
        lookup_D1_D = lookup[(D - 1, D)]::SparseOp{D - 1,D,One}
        lookup_R_D1 = lookup[(R, D - 1)]::SparseOp{R,D - 1,One}
        lookup_D2_D1 = lookup[(D - 2, D - 1)]::SparseOp{D - 2,D - 1,One}
        lookup_R_D2 = lookup[(R, D - 2)]::SparseOp{R,D - 2,One}
        dualcoords_R = dualcoords[R]
        dualcoords_D = dualcoords[D]
        dualcoords_D1 = dualcoords[D - 1]
        dualcoords_D2 = dualcoords[D - 2]

        # Loop over all `R`-forms
        @inbounds for i in 1:size(simplices, 2)
            bci = dualcoords_R[i]

            vol = S(0)
            # Loop over all neighbouring `D`-forms
            for j in sparse_column_rows(lookup_D_R, i)
                bcj = dualcoords_D[j]

                # Loop over all contained `D-1`-forms that contain `i`
                for k in sparse_column_rows(lookup_D1_D, j)
                    if i ∈ sparse_column_rows(lookup_R_D1, k)
                        bck = dualcoords_D1[k]

                        # Loop over all contained `D-2`-forms that contain `i`
                        for l in sparse_column_rows(lookup_D2_D1, k)
                            if i ∈ sparse_column_rows(lookup_R_D2, l)
                                bcl = dualcoords_D2[l]

                                vol += volume(SVector(bci, bcj, bck, bcl))
                            end
                        end
                    end
                end
            end
            dualvolumes[i] = vol
        end

    elseif R == D - 4
        lookup_D_R = lookup[(D, R)]::SparseOp{D,R,One}
        lookup_D1_D = lookup[(D - 1, D)]::SparseOp{D - 1,D,One}
        lookup_R_D1 = lookup[(R, D - 1)]::SparseOp{R,D - 1,One}
        lookup_D2_D1 = lookup[(D - 2, D - 1)]::SparseOp{D - 2,D - 1,One}
        lookup_R_D2 = lookup[(R, D - 2)]::SparseOp{R,D - 2,One}
        lookup_D3_D2 = lookup[(D - 3, D - 2)]::SparseOp{D - 3,D - 2,One}
        lookup_R_D3 = lookup[(R, D - 3)]::SparseOp{R,D - 3,One}
        dualcoords_R = dualcoords[R]
        dualcoords_D = dualcoords[D]
        dualcoords_D1 = dualcoords[D - 1]
        dualcoords_D2 = dualcoords[D - 2]
        dualcoords_D3 = dualcoords[D - 3]

        # Loop over all `R`-forms
        @inbounds for i in 1:size(simplices, 2)
            bci = dualcoords_R[i]

            vol = S(0)
            # Loop over all neighbouring `D`-forms
            for j in sparse_column_rows(lookup_D_R, i)
                bcj = dualcoords_D[j]

                # Loop over all contained `D-1`-forms that contain `i`
                for k in sparse_column_rows(lookup_D1_D, j)
                    if i ∈ sparse_column_rows(lookup_R_D1, k)
                        bck = dualcoords_D1[k]

                        # Loop over all contained `D-2`-forms that contain `i`
                        for l in sparse_column_rows(lookup_D2_D1, k)
                            if i ∈ sparse_column_rows(lookup_R_D2, l)
                                bcl = dualcoords_D2[l]

                                # Loop over all contained `D-3`-forms
                                # that contain `i`
                                for m in sparse_column_rows(lookup_D3_D2, l)
                                    if i ∈ sparse_column_rows(lookup_R_D3, m)
                                        bcm = dualcoords_D3[m]

                                        vol += volume(SVector(bci, bcj, bck,
                                                              bcl, bcm))
                                    end
                                end
                            end
                        end
                    end
                end
            end
            dualvolumes[i] = vol
        end

    elseif R == D - 5
        lookup_D_R = lookup[(D, R)]::SparseOp{D,R,One}
        lookup_D1_D = lookup[(D - 1, D)]::SparseOp{D - 1,D,One}
        lookup_R_D1 = lookup[(R, D - 1)]::SparseOp{R,D - 1,One}
        lookup_D2_D1 = lookup[(D - 2, D - 1)]::SparseOp{D - 2,D - 1,One}
        lookup_R_D2 = lookup[(R, D - 2)]::SparseOp{R,D - 2,One}
        lookup_D3_D2 = lookup[(D - 3, D - 2)]::SparseOp{D - 3,D - 2,One}
        lookup_R_D3 = lookup[(R, D - 3)]::SparseOp{R,D - 3,One}
        lookup_D4_D3 = lookup[(D - 4, D - 3)]::SparseOp{D - 4,D - 3,One}
        lookup_R_D4 = lookup[(R, D - 4)]::SparseOp{R,D - 4,One}
        dualcoords_R = dualcoords[R]
        dualcoords_D = dualcoords[D]
        dualcoords_D1 = dualcoords[D - 1]
        dualcoords_D2 = dualcoords[D - 2]
        dualcoords_D3 = dualcoords[D - 3]
        dualcoords_D4 = dualcoords[D - 4]

        # Loop over all `R`-forms
        @inbounds for i in 1:size(simplices, 2)
            bci = dualcoords_R[i]

            vol = S(0)
            # Loop over all neighbouring `D`-forms
            for j in sparse_column_rows(lookup_D_R, i)
                bcj = dualcoords_D[j]

                # Loop over all contained `D-1`-forms that contain `i`
                for k in sparse_column_rows(lookup_D1_D, j)
                    if i ∈ sparse_column_rows(lookup_R_D1, k)
                        bck = dualcoords_D1[k]

                        # Loop over all contained `D-2`-forms that contain `i`
                        for l in sparse_column_rows(lookup_D2_D1, k)
                            if i ∈ sparse_column_rows(lookup_R_D2, l)
                                bcl = dualcoords_D2[l]

                                # Loop over all contained `D-3`-forms
                                # that contain `i`
                                for m in sparse_column_rows(lookup_D3_D2, l)
                                    if i ∈ sparse_column_rows(lookup_R_D3, m)
                                        bcm = dualcoords_D3[m]

                                        # Loop over all contained
                                        # `D-4`-forms that contain `i`
                                        for n in
                                            sparse_column_rows(lookup_D4_D3, m)
                                            if i ∈
                                               sparse_column_rows(lookup_R_D4,
                                                                  n)
                                                bcn = dualcoords_D3[n]

                                                vol += volume(SVector(bci, bcj,
                                                                      bck, bcl,
                                                                      bcm, bcn))
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            dualvolumes[i] = vol
        end

    else
        @assert false
    end

    return dualvolumes
end

"""
Calculate circumcentric dual volumes

See [1198555.1198667, section 6.2.1]
"""
function calc_dualvolumes(::Val{CircumcentricDuals}, ::Val{D},
                          simplices::SparseOp{0,R,One},
                          simplices1::SparseOp{0,R1,One},
                          parents::SparseOp{R1,R,One},
                          coords::Vector{SVector{C,S}}, volumes::Vector{S},
                          dualcoords::Vector{SVector{C,S}},
                          dualcoords1::Vector{SVector{C,S}},
                          dualvolumes1::Vector{S}) where {dualkind,D,R,R1,C,S}
    R::Int
    R1::Int
    C::Int
    @assert 0 ≤ R ≤ R1 ≤ D
    @assert R1 == R + 1
    nvertices, nsimplices = size(simplices)
    dualvolumes = Array{S}(undef, nsimplices)
    # Loop over all `R`-simplices
    @inbounds for i in 1:nsimplices
        si = sparse_column_rows(simplices, i)
        @assert length(si) == R + 1
        xsi = SVector{R + 1}(Form{C,1}(coords[i]) for i in si)
        bci = barycentre(xsi)::Form{C,1,S}
        cci = Form{C,1,S}(dualcoords[i])

        voli = zero(S)
        # Loop over all neighbouring `R+1`-simplices
        for j in sparse_column_rows(parents, i)
            sj = sparse_column_rows(simplices1, j)
            @assert length(sj) == R + 2

            xsj = SVector{R + 2}(Form{C,1}(coords[j]) for j in sj)
            bcj = barycentre(xsj)::Form{C,1,S}
            ccj = Form{C,1,S}(dualcoords1[j])

            ysi = map(y -> y - xsi[1], deleteat(xsi, 1))   # R
            ysj = map(y -> y - xsj[1], deleteat(xsj, 1))   # R+1
            @assert !isempty(ysj)
            ni = ∧(ysj) ⋅ ∧(ysi)   # 1
            ni::Form{C,1,S}
            qsi = map(y -> norm(ni ⋅ y) < 10 * eps(S), ysi)
            @assert all(qsi)
            lni = norm(ni)
            @assert lni > 0
            nni = ni / lni

            s0 = (bcj - xsi[1]) ⋅ nni
            s0::Form{C,0,S}
            s = bitsign(signbit(s0[]))
            h0 = (ccj - cci) ⋅ nni
            h0::Form{C,0,S}
            h = s * h0[]

            b = dualvolumes1[j]
            volj = b * h

            voli += volj
        end

        dualvolumes[i] = voli / (D - R)
    end
    return dualvolumes
end

end
