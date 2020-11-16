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

const VecDict{K,T} = Dict{K,IDVector{<:Any,T}} where {T}
const OpDict{K,T} = Dict{K,SparseOp{<:Any,<:Any,T}} where {K,T}

export Manifold
"""
A discrete manifold 
"""
mutable struct Manifold{D,C,S}
    name::String

    signature::Int              # [+1, -1]

    dualkind::DualKind
    use_weighted_duals::Bool

    # TODO: Implement also a "cube complex" representation

    # If `simplices[R][i,j]` is present, then vertex `i` is part of
    # the `R`-simplex `j`. `R ∈ 0:D`. We could omit `R=0`.
    _simplices::OpDict{Int,One}
    # simplices::Dict{Int,Array{Int,2}}

    # The boundary ∂ of `0`-forms vanishes and is not stored. If
    # `boundaries[R][i,j] = s`, then `R`-simplex `j` has
    # `(R-1)`-simplex `i` as boundary with orientation `s`. `R ∈ 1:D`.
    _boundaries::OpDict{Int,Int8}
    _isboundary::OpDict{Int,One}

    # Lookup tables from `Ri`-simplices to `Rj`-simplices. If
    # `lookup[(Ri,Rj)][i,j]` is present, then `Rj`-simplex `j`
    # contains `Ri`-simplex `i`. `Ri ∈ 0:D, Rj ∈ 0:D`. Many of these
    # are trivial and could be omitted.
    _lookup::OpDict{Tuple{Int,Int},One}
    # lookup::Dict{Tuple{Int,Int},Array{Int,2}}

    # coords::Array{S,2}
    _coords::VecDict{Int,SVector{C,S}}
    _volumes::VecDict{Int,S}

    # Vertex weights [arXiv:math/0508188], [DOI:10.1145/2602143]
    _weights::IDVector{0,S}

    # Coordinates of vertices of dual grid, i.e.
    # barycentres/circumcentres of primal top-simplices
    # dualcoords::Array{S,2}
    _dualcoords::VecDict{Int,SVector{C,S}}
    # `dualvolumes[R]` are the volumes of the simplices dual to the
    # primal `R`-simplices
    _dualvolumes::VecDict{Int,S}

    # Nearest neighbour tree for simplex vertices
    _simplex_tree::Union{Nothing,KDTree{SVector{C,S},Euclidean,S}}

    function Manifold{D,C,S}(name::String, signature::Int, dualkind::DualKind,
                             use_weighted_duals::Bool,
                             simplices::OpDict{Int,One},
                             boundaries::OpDict{Int,Int8},
                             coords0::IDVector{0,SVector{C,S}},
                             weights::IDVector{0,S}) where {D,C,S}
        D::Int
        @assert 0 ≤ D ≤ C
        mfd = new{D,C,S}(name, signature, dualkind, use_weighted_duals,
                         simplices, boundaries, OpDict{Int,One}(),
                         OpDict{Tuple{Int,Int},One}(),
                         VecDict{Int,SVector{C,S}}(0 => coords0),
                         VecDict{Int,S}(), weights, VecDict{Int,SVector{C,S}}(),
                         VecDict{Int,S}(), nothing)
        #TODO @assert invariant(mfd)
        return mfd
    end
end

function Manifold(name::String, signature::Int, dualkind::DualKind,
                  use_weighted_duals::Bool, simplices::OpDict{Int,One},
                  boundaries::OpDict{Int,Int8},
                  coords0::IDVector{0,SVector{C,S}},
                  weights::IDVector{0,S}) where {C,S}
    D = maximum(keys(simplices))
    return Manifold{D,C,S}(name, signature, dualkind, use_weighted_duals,
                           simplices, boundaries, coords0, weights)
end

################################################################################

function Base.show(io::IO, mfd::Manifold{D}) where {D}
    println(io)
    println(io, "Manifold{$D}(")
    println(io, "    name=$(mfd.name)")
    for R in 0:D
        println(io, "    simplices[$R]=$(mfd._simplices[R])")
    end
    for R in 1:D
        println(io, "    boundaries[$R]=$(mfd._boundaries[R])")
    end
    for R in 0:(D - 1)
        if haskey(mfd._isboundary, R)
            println(io, "    isboundary[$R]=$(mfd._isboundary[R])")
        end
    end
    for R in 0:D
        if haskey(mfd._coords, R)
            println(io, "    coords[$R]=$(mfd._coords[R])")
        end
    end
    for R in 0:D
        if haskey(mfd._volumes, R)
            println(io, "    volumes[$R]=$(mfd._volumes[R])")
        end
    end
    println(io, "    weights=$(mfd._weights)")
    for R in 0:D
        if haskey(mfd._dualcoords, R)
            println(io, "    dualcoords[$R]=$(mfd._dualcoords[R])")
        end
    end
    for R in 0:D
        if haskey(mfd._dualvolumes, R)
            println(io, "    dualvolumes[$R]=$(mfd._dualvolumes[R])")
        end
    end
    return print(io, ")")
end

function Defs.invariant(mfd::Manifold{D,C})::Bool where {D,C}
    D ≥ 0 || (@assert false; return false)
    C ≥ D || (@assert false; return false)

    # Check simplices
    Set(keys(mfd._simplices)) == Set(0:D) || (@assert false; return false)
    for R in 0:D
        simplices = get_simplices(mfd, R)
        size(simplices) == (nsimplices(mfd, 0), nsimplices(mfd, R)) ||
            (@assert false; return false)
        for j in axes(simplices, 2)
            sj = sparse_column_rows(simplices, j)
            length(sj) == R + 1 || (@assert false; return false)
        end
    end

    # Check boundaries
    Set(keys(mfd._boundaries)) == Set(1:D) || (@assert false; return false)
    for R in 1:D
        boundaries = get_boundaries(mfd, R)
        size(boundaries) == (nsimplices(mfd, R - 1), nsimplices(mfd, R)) ||
            (@assert false; return false)
        simplicesR = get_simplices(mfd, R)
        simplicesR1 = get_simplices(mfd, R - 1)
        for j in axes(boundaries, 2) # R-simplex
            vj = sparse_column_rows(simplicesR, j)
            sj = sparse_column(boundaries, j)
            for (i, p) in sj    # (R-1)-simplex
                si = sparse_column_rows(simplicesR1, i)
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
    # Set(keys(mfd._lookup)) == Set((Ri, Rj) for Ri in 0:D for Rj in 0:D) ||
    #     (@assert false; return false)
    # for Ri in 0:D, Rj in 0:D
    #     lookup = mfd._lookup[(Ri, Rj)]::SparseOp{Ri,Rj,One}
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
    #         mfd._lookup[(Ri, Rj)] == mfd._lookup[(Rj, Ri)]' ||
    #             (@assert false; return false)
    #     end
    # end

    # Set(keys(mfd._coords)) == Set(0:D) || (@assert false; return false)
    # for R in 0:D
    #     length(mfd._coords[R]) == nsimplices(mfd, R) ||
    #         (@assert false; return false)
    # end
    # 
    # Set(keys(mfd._volumes)) == Set(0:D) || (@assert false; return false)
    # for R in 0:D
    #     length(mfd._volumes[R]) == nsimplices(mfd, R) ||
    #         (@assert false; return false)
    # end

    length(mfd._weights) == nsimplices(mfd, 0) || (@assert false; return false)

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
    return mfd1.signature == mfd2.signature &&
           mfd1.dualkind == mfd2.dualkind &&
           mfd1.use_weighted_duals == mfd2.use_weighted_duals &&
           mfd1._simplices[D] == mfd2._simplices[D] &&
           mfd1._coords[0] == mfd2._coords[0] &&
           mfd1._weights == mfd2._weights
end
function Base.isequal(mfd1::Manifold{D,C},
                      mfd2::Manifold{D,C})::Bool where {D,C}
    return
    return isequal(mfd1.signature, mfd2.signature) &&
           isequal(mfd1.dualkind, mfd2.dualkind) &&
           isequal(mfd1.use_weighted_duals, mfd2.use_weighted_duals) &&
           isequal(mfd1._simplices[D], mfd2._simplices[D]) &&
           isequal(mfd1._coords[0], mfd2._coords[0]) &&
           isequal(mfd1._weights, mfd2._weights)
end
function Base.hash(mfd::Manifold{D,C}, h::UInt) where {D,C}
    return hash(0x4d34f4ae,
                hash(D,
                     hash(C,
                          hash(mfd.signature,
                               hash(mfd.dualkind,
                                    hash(mfd.use_weighted_duals,
                                         hash(mfd._simplices[D],
                                              hash(mfd._coords[0],
                                                   hash(mfd._weights[0], h)))))))))
end

Base.ndims(::Manifold{D}) where {D} = D

export nsimplices
function nsimplices(mfd::Manifold, ::Val{R}) where {R}
    return size(get_simplices(Val(R), mfd), 2)
end
nsimplices(mfd::Manifold, R::Integer) = size(get_simplices(mfd, R), 2)

export random_point
"""
Return random point in manifold
"""
function random_point(::Val{R}, mfd::Manifold{D,C,S}) where {D,C,R,S}
    @assert 0 ≤ R ≤ D ≤ C
    C == 0 && return SVector{C,S}()
    N = R + 1
    # Choose simplex
    i = ID{R}(rand(1:nsimplices(mfd, R)))
    si = sparse_column_rows(get_simplices(mfd, R), i)
    @assert length(si) == N
    # Choose point in simplex
    λ = abs.(randn(SVector{N,S}))
    λ /= sum(λ)
    x = sum(λ[n] * get_coords(mfd)[si[n]] for n in 1:N)
    return x::SVector{C,S}
end

################################################################################

# Outer constructor

function Manifold(name::String, simplicesD::SparseOp{0,D,One},
                  coords0::IDVector{0,SVector{C,S}}, weights::IDVector{0,S};
                  signature::Int=+1, dualkind::DualKind=CircumcentricDuals,
                  optimize_mesh::Bool=true,
                  use_weighted_duals::Bool=true) where {D,C,S}
    @assert 0 ≤ D ≤ C

    nvertices, nsimplices = size(simplicesD)
    @assert length(coords0) == nvertices
    @assert length(weights) == nvertices

    # Calculate lower-Dimensional simplices
    simplices = OpDict{Int,One}(D => simplicesD)
    boundaries = OpDict{Int,Int8}()
    for R in D:-1:1
        facesR, boundariesR = calc_faces_boundaries(simplices[R])
        simplices[R - 1] = facesR
        boundaries[R] = boundariesR
    end

    if optimize_mesh && use_weighted_duals
        @assert dualkind == CircumcentricDuals
        # Optimize weights

        # Calculate mask for boundary vertices (which must not be moved)
        if D > 0
            boundary_faces = IDVector{D - 1}(zeros(Int8,
                                                   size(boundaries[D], 1)))
            for j in axes(boundaries[D], 2)
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
            dof = IDVector{0}(ones(SMatrix{C,C,S}, length(coords0)))
            @assert length(boundary_faces) == size(simplices[D - 1], 2)
            for j in axes(simplices[D - 1], 2)
                if isodd(boundary_faces[j])
                    sj = sparse_column_rows(simplices[D - 1], j)
                    sj = SVector{D,ID{0}}(sj[n] for n in 1:D)
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
            dof = IDVector{0}(zeros(SMatrix{C,C,S}, length(coords0)))
        end
        dof::IDVector{0,SMatrix{C,C,S}}
        coords0, weights = optimize_mesh1(Val(dualkind), Val(D), simplices,
                                          coords0, dof, weights)
    end

    # TODO: Move this to invariant, maybe add `haskey` for dualcoords
    # if D > 0
    #     check_delaunay(simplices[D], lookup[(D - 1, D)], lookup[(D, D - 1)],
    #                    coords[0], dualcoords[D])
    # end

    # Create D-manifold
    return Manifold(name, signature, dualkind, use_weighted_duals, simplices,
                    boundaries, coords0, weights)
end

################################################################################

export get_simplices
@inline function get_simplices(::Val{R}, mfd::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R ≤ D
    return mfd._simplices[R]::SparseOp{0,R,One}
end
@inline function get_simplices(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    return get_simplices(Val(R), mfd)
end

export get_boundaries
@inline function get_boundaries(::Val{R}, mfd::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 < R ≤ D
    return mfd._boundaries[R]::SparseOp{R - 1,R,Int8}
end
@inline function get_boundaries(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    return get_boundaries(Val(R), mfd)
end

export get_lookup
@inline function get_lookup(::Val{Ri}, ::Val{Rj},
                            mfd::Manifold{D,C,S}) where {Ri,Rj,D,C,S}
    @assert 0 ≤ Rj ≤ D
    @assert 0 ≤ Ri ≤ D
    !haskey(mfd._lookup, (Ri, Rj)) && calc_lookup!(mfd, Ri, Rj)
    return mfd._lookup[(Ri, Rj)]::SparseOp{Ri,Rj,One}
end
@inline function get_lookup(mfd::Manifold{D,C,S}, Ri::Int,
                            Rj::Int) where {D,C,S}
    return get_lookup(Val(Ri), Val(Rj), mfd)
end

function calc_lookup!(mfd::Manifold{D,C,S}, Ri::Int, Rj::Int) where {D,C,S}
    @assert 0 ≤ Rj ≤ D
    @assert 0 ≤ Ri ≤ D
    @assert !haskey(mfd._lookup, (Ri, Rj))

    sizei = nsimplices(mfd, Ri)
    sizej = nsimplices(mfd, Rj)

    if Ri == 0
        # Simplex definitions
        mfd._lookup[(Ri, Rj)] = get_simplices(mfd, Rj)
    elseif Ri == Rj
        # Identity
        mfd._lookup[(Ri, Rj)] = SparseOp{Ri,Rj}(sparse(1:sizei, 1:sizej,
                                                       fill(One(), sizei)))
    elseif Ri == Rj - 1
        # Absolute value of boundaries
        mfd._lookup[(Ri, Rj)] = map(x -> One(x ≠ 0), get_boundaries(mfd, Rj))
    elseif Ri < Rj - 1
        # Chain of two lookup tables
        Rk = Rj - 1
        mfd._lookup[(Ri, Rj)] = map(x -> One(x ≠ 0),
                                    get_lookup(mfd, Ri, Rk) *
                                    get_lookup(mfd, Rk, Rj))
    else
        # Transpose
        mfd._lookup[(Ri, Rj)] = get_lookup(mfd, Rj, Ri)'
    end

    return nothing
end

export get_isboundary
@inline function get_isboundary(::Val{R}, mfd::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R < D
    !haskey(mfd._isboundary, R) && calc_isboundary!(mfd, R)
    return mfd._isboundary[R]::SparseOp{R,R,One}
end
@inline function get_isboundary(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    return get_isboundary(Val(R), mfd)
end

function calc_isboundary!(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    @assert 0 ≤ R < D
    @assert !haskey(mfd._isboundary, R)

    I = Int[]
    J = Int[]
    V = One[]
    nelts = nsimplices(mfd, R)
    if R == D - 1
        boundariesD = get_boundaries(mfd, D)
        isbndface = IDVector{D - 1}(zeros(Bool, size(boundariesD, 1)))
        for j in axes(boundariesD, 2)
            for i in sparse_column_rows(boundariesD, j)
                isbndface[i] = !isbndface[i]
            end
        end
        for (i, b) in enumerate(isbndface)
            if b
                push!(I, Int(i))
                push!(J, Int(i))
                push!(V, One())
            end
        end
    else
        lookup = get_lookup(mfd, R, D - 1)
        for j0 in findnz(get_isboundary(mfd, D - 1).op)[1]
            j = ID{D - 1}(j0)
            for i in sparse_column_rows(lookup, j)
                push!(I, Int(i))
                push!(J, Int(i))
                push!(V, One())
            end
        end
    end
    mfd._isboundary[R] = SparseOp{R,R}(sparse(I, J, V, nelts, nelts, max))

    return nothing
end

export get_coords
@inline get_coords(mfd::Manifold) = get_coords(Val(0), mfd)
@inline function get_coords(::Val{R}, mfd::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R ≤ D
    !haskey(mfd._coords, R) && calc_coords!(mfd, R)
    return mfd._coords[R]::IDVector{R,SVector{C,S}}
end
@inline function get_coords(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    return get_coords(Val(R), mfd)
end

function calc_coords!(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    @assert 0 ≤ R ≤ D
    @assert !haskey(mfd._coords, R)
    mfd._coords[R] = calc_coords(get_simplices(mfd, R), get_coords(mfd))
    return nothing
end

export get_volumes
@inline function get_volumes(::Val{R}, mfd::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R ≤ D
    !haskey(mfd._volumes, R) && calc_volumes!(mfd, R)
    return mfd._volumes[R]::IDVector{R,S}
end
@inline function get_volumes(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    return get_volumes(Val(R), mfd)
end

function calc_volumes!(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    @assert 0 ≤ R ≤ D
    @assert !haskey(mfd._volumes, R)
    mfd._volumes[R] = calc_volumes(get_simplices(mfd, R), get_coords(mfd),
                                   mfd.signature)

    # Ensure that all volumes are strictly positive
    allpositive = all(>(0), mfd._volumes[R])
    if !allpositive
        @warn "Not all  $R-volumes are strictly positive"
        @show count(≤(0), mfd._volumes[R])
        @show findmin(mfd._volumes[R])
        @show get_coords(mfd, R)[ID{R}(findmin(mfd._volumes[R])[2])]
    end

    return nothing
end

export get_weights
@inline function get_weights(mfd::Manifold{D,C,S}) where {D,C,S}
    return mfd._weights::IDVector{0,S}
end

export get_dualcoords
@inline function get_dualcoords(::Val{R}, mfd::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R ≤ D
    !haskey(mfd._dualcoords, R) && calc_dualcoords!(mfd, R)
    return mfd._dualcoords[R]::IDVector{R,SVector{C,S}}
end
@inline function get_dualcoords(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    return get_dualcoords(Val(R), mfd)
end

function calc_dualcoords!(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    @assert 0 ≤ R ≤ D
    @assert !haskey(mfd._dualcoords, R)
    if mfd.use_weighted_duals
        mfd._dualcoords[R] = calc_dualcoords(Val(mfd.dualkind),
                                             get_simplices(mfd, R),
                                             get_coords(mfd), get_weights(mfd))
    else
        mfd._dualcoords[R] = calc_dualcoords(Val(mfd.dualkind),
                                             get_simplices(mfd, R),
                                             get_coords(mfd))
    end
    return nothing
end

export get_dualvolumes
@inline function get_dualvolumes(::Val{R}, mfd::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R ≤ D
    !haskey(mfd._dualvolumes, R) && calc_dualvolumes!(mfd, R)
    return mfd._dualvolumes[R]::IDVector{R,S}
end
@inline function get_dualvolumes(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    return get_dualvolumes(Val(R), mfd)
end

function calc_dualvolumes!(mfd::Manifold{D,C,S}, R::Int) where {D,C,S}
    @assert 0 ≤ R ≤ D
    @assert !haskey(mfd._dualvolumes, R)
    if mfd.dualkind == BarycentricDuals
        mfd._dualvolumes[R] = calc_dualvolumes(Val(mfd.dualkind), Val(D),
                                               Val(R), get_simplices(mfd, R),
                                               mfd.lookup, get_coords(mfd),
                                               get_volumes(mfd, D))
        @assert all(x -> x != 0 && isfinite(x), mfd._dualvolumes[R])
    elseif mfd.dualkind == CircumcentricDuals
        if R == D
            mfd._dualvolumes[D] = IDVector{D}(fill(S(1), nsimplices(mfd, D)))
        else
            mfd._dualvolumes[R] = calc_dualvolumes(Val(mfd.dualkind), Val(D),
                                                   get_simplices(mfd, R),
                                                   get_simplices(mfd, R + 1),
                                                   get_lookup(mfd, R + 1, R),
                                                   get_coords(mfd),
                                                   get_dualcoords(mfd, R),
                                                   get_dualcoords(mfd, R + 1),
                                                   get_dualvolumes(mfd, R + 1))
        end

        # # TODO: These should be equivalent
        # dualvolsR = calc_dualvolumes(Val(BarycentricDuals), Val(R), mfd)
        # if !(mfd._dualvolumes[R] ≈ dualvolsR)
        #     @show D R mfd._dualvolumes[R] dualvolsR
        # end
        # @assert mfd._dualvolumes[R] ≈ dualvolsR

    else
        @assert false
    end

    # Ensure that all dual volumes are strictly positive
    allpositive = all(>(0), mfd._dualvolumes[R])
    if !allpositive
        @warn "Not all dual $R-volumes are strictly positive"
        @show count(≤(0), mfd._dualvolumes[R])
        @show findmin(mfd._dualvolumes[R])
        @show get_dualcoords(mfd, R)[ID{R}(findmin(mfd._dualvolumes[R])[2])]
    end

    return nothing
end

export get_simplex_tree
@inline function get_simplex_tree(mfd::Manifold{D,C,S}) where {D,C,S}
    mfd._simplex_tree === nothing && calc_simplex_tree!(mfd)
    return mfd._simplex_tree::KDTree{SVector{C,S},Euclidean,S}
end

function calc_simplex_tree!(mfd::Manifold{D,C,S}) where {D,C,S}
    @assert mfd._simplex_tree === nothing
    mfd._simplex_tree = KDTree(get_coords(mfd, D).vec)
    return nothing
end

################################################################################

struct Face{D,N}
    vertices::SVector{N,ID{0}}
    parent::ID{D}
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
    facelist = Face{D,D}[]
    for j in axes(simplices, 2)
        ks = SVector{D + 1,ID{0}}(sparse_column_rows(simplices, j)[n]
                                  for n in 1:(D + 1))
        for n in 1:(D + 1)
            # Leave out vertex n
            ls = deleteat(ks, n)
            p = bitsign(isodd(n - 1))
            push!(facelist, Face{D,D}(ls, j, p))
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
        if f.vertices ≠ oldvertices
            oldvertices = f.vertices
            # We found a new face
            nfaces += 1
            for i in f.vertices
                push!(fI, Int(i))
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
                        coords::IDVector{0,SVector{C,S}},
                        dof::IDVector{0,SMatrix{C,C,S}},
                        weights::IDVector{0,S}) where {D,C,S}
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
                for i in axes(new_coords, 1)
                    # nnzev = count(isapprox(1), eigvals(dof[i]))
                    # if nnzev ≤ D′
                    # new_coords[i] += α * dof[i] * shift_cs[i]
                    new_weights[i] += α * shift_ws[i]
                    # end
                end
            end
            avg_weight = sum(new_weights) / length(new_weights)
            for i in axes(new_weights, 1)
                new_weights[i] -= avg_weight
            end
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
                     coords0::IDVector{0,SVector{C,S}}) where {D,C,S}
    nvertices, nsimplices = size(simplices)
    D == 0 && return coords0
    coords = IDVector{D}(Array{SVector{C,S}}(undef, nsimplices))
    @inbounds for i in axes(coords, 1)
        si = sparse_column_rows(simplices, i)
        si = SVector{D + 1}(si[n] for n in 1:(D + 1))
        xs = SVector{D + 1}(coords0[i] for i in si)
        coords[i] = barycentre(xs)
    end
    return coords
end

"""
Calculate volumes
"""
function calc_volumes(simplices::SparseOp{0,D,One},
                      coords::IDVector{0,SVector{C,S}},
                      signature::Int=1) where {D,C,S}
    nvertices, nsimplices = size(simplices)
    D == 0 && return IDVector{D}(fill(one(S), nvertices))
    volumes = IDVector{D}(Array{S}(undef, nsimplices))
    @inbounds for i in axes(volumes, 1)
        si = sparse_column_rows(simplices, i)
        si = SVector{D + 1}(si[n] for n in 1:(D + 1))
        xs = SVector{D + 1}(Form{C,1}(coords[i]) for i in si)
        # volumes[i] = volume(xs)
        vol = volume(xs)
        svol = signed_volume(xs)
        svols = signed_volume(xs, signature)
        if !(abs(svol) ≈ vol)
            @show D C i vol svol
        end
        @assert abs(svol) ≈ vol
        volumes[i] = svols * sign(svol)
    end
    return volumes
end

"""
Calculate barycentric dual coordinates
"""
function calc_dualcoords(::Val{BarycentricDuals}, simplices::SparseOp{0,D,One},
                         coords::IDVector{0,SVector{C,S}}) where {D,C,S}
    D == 0 && return coords
    nsimplices = size(simplices, 2)
    dualcoords = IDVector{D}(Array{SVector{C,S}}(undef, nsimplices))
    @inbounds for i in axes(dualcoords, 1)
        si = sparse_column_rows(simplices, i)
        si = SVector{D + 1}(si[n] for n in 1:(D + 1))
        xs = SVector{D + 1}(Form{C,1}(coords[i]) for i in si)
        dualcoords[i] = barycentre(xs)
    end
    return dualcoords
end

"""
Calculate circumcentric dual coordinates
"""
function calc_dualcoords(::Val{CircumcentricDuals},
                         simplices::SparseOp{0,D,One},
                         coords::IDVector{0,SVector{C,S}}) where {D,C,S}
    nvertices, nsimplices = size(simplices)
    @assert length(coords) == nvertices
    D == 0 && return coords
    dualcoords = IDVector{D}(Array{SVector{C,S}}(undef, nsimplices))
    @inbounds for i in axes(dualcoords, 1)
        si = sparse_column_rows(simplices, i)
        si = SVector{D + 1}(si[n] for n in 1:(D + 1))
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
                         coords::IDVector{0,SVector{C,S}},
                         weights::IDVector{0,S}) where {D,C,S}
    nvertices, nsimplices = size(simplices)
    @assert length(coords) == nvertices
    @assert length(weights) == nvertices
    D == 0 && return coords
    dualcoords = IDVector{D}(Array{SVector{C,S}}(undef, nsimplices))
    @inbounds for i in axes(dualcoords, 1)
        si = sparse_column_rows(simplices, i)
        si = SVector{D + 1}(si[n] for n in 1:(D + 1))
        xs = SVector{D + 1}(Form{C,1}(coords[i]) for i in si)
        ws = SVector{D + 1}(Form{C,0}((weights[i],)) for i in si)
        dualcoords[i] = circumcentre(xs, ws)
    end
    return dualcoords
end

"""
Calculate barycentric dual volumes
"""
function calc_dualvolumes(::Val{BarycentricDuals}, ::Val{R},
                          manifold::Manifold{D,C,S}) where {R,D,C,S}
    R::Int
    D::Int
    C::Int
    @assert 0 ≤ R ≤ D ≤ C

    simplices_R = get_simplices(manifold, R)

    dualvolumes = IDVector{R}(Array{S}(undef, size(simplices_R, 2)))

    if R == D
        @inbounds for i in axes(dualvolumes, 1)
            dualvolumes[i] = 1
        end

    elseif false && R == 0
        lookup_D_R = get_lookup(manifold, D, R)

        volumes_D = get_volumes(manifold, D)
        for i in axes(dualvolumes, 1)
            vol = S(0)
            for j in sparse_column_rows(lookup_D_R, i)
                vol += volumes_D[j]
            end
            dualvolumes[i] = vol / (D + 1)
        end

    elseif R == D - 1
        lookup_D_R = get_lookup(manifold, D, R)
        dualcoords_R = get_dualcoords(manifold, R)
        dualcoords_D = get_dualcoords(manifold, D)

        @inbounds for i in axes(dualvolumes, 1)
            bci = dualcoords_R[i]

            vol = S(0)
            for j in sparse_column_rows(lookup_D_R, i)
                bcj = dualcoords_D[j]

                vol += volume(SVector(bci, bcj))
            end
            dualvolumes[i] = vol
        end

    elseif R == D - 2
        lookup_D_R = get_lookup(manifold, D, R)
        lookup_D1_D = get_lookup(manifold, D - 1, D)
        lookup_R_D1 = get_lookup(manifold, R, D - 1)
        dualcoords_R = get_dualcoords(manifold, R)
        dualcoords_D = get_dualcoords(manifold, D)
        dualcoords_D1 = get_dualcoords(manifold, D - 1)

        # Loop over all `R`-forms
        @inbounds for i in axes(dualvolumes, 1)
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
        lookup_D_R = get_lookup(manifold, D, R)
        lookup_D1_D = get_lookup(manifold, D - 1, D)
        lookup_R_D1 = get_lookup(manifold, R, D - 1)
        lookup_D2_D1 = get_lookup(manifold, D - 2, D - 1)
        lookup_R_D2 = get_lookup(manifold, R, D - 2)
        dualcoords_R = get_dualcoords(manifold, R)
        dualcoords_D = get_dualcoords(manifold, D)
        dualcoords_D1 = get_dualcoords(manifold, D - 1)
        dualcoords_D2 = get_dualcoords(manifold, D - 2)

        # Loop over all `R`-forms
        @inbounds for i in axes(dualvolumes, 1)
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
        lookup_D_R = get_lookup(manifold, D, R)
        lookup_D1_D = get_lookup(manifold, D - 1, D)
        lookup_R_D1 = get_lookup(manifold, R, D - 1)
        lookup_D2_D1 = get_lookup(manifold, D - 2, D - 1)
        lookup_R_D2 = get_lookup(manifold, R, D - 2)
        lookup_D3_D2 = get_lookup(manifold, D - 3, D - 2)
        lookup_R_D3 = get_lookup(manifold, R, D - 3)
        dualcoords_R = get_dualcoords(manifold, R)
        dualcoords_D = get_dualcoords(manifold, D)
        dualcoords_D1 = get_dualcoords(manifold, D - 1)
        dualcoords_D2 = get_dualcoords(manifold, D - 2)
        dualcoords_D3 = get_dualcoords(manifold, D - 3)

        # Loop over all `R`-forms
        @inbounds for i in axes(dualvolumes, 1)
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
        lookup_D_R = get_lookup(manifold, D, R)
        lookup_D1_D = get_lookup(manifold, D - 1, D)
        lookup_R_D1 = get_lookup(manifold, R, D - 1)
        lookup_D2_D1 = get_lookup(manifold, D - 2, D - 1)
        lookup_R_D2 = get_lookup(manifold, R, D - 2)
        lookup_D3_D2 = get_lookup(manifold, D - 3, D - 2)
        lookup_R_D3 = get_lookup(manifold, R, D - 3)
        lookup_D4_D3 = get_lookup(manifold, D - 4, D - 3)
        lookup_R_D4 = get_lookup(manifold, R, D - 4)
        dualcoords_R = get_dualcoords(manifold, R)
        dualcoords_D = get_dualcoords(manifold, D)
        dualcoords_D1 = get_dualcoords(manifold, D - 1)
        dualcoords_D2 = get_dualcoords(manifold, D - 2)
        dualcoords_D3 = get_dualcoords(manifold, D - 3)
        dualcoords_D4 = get_dualcoords(manifold, D - 4)

        # Loop over all `R`-forms
        @inbounds for i in axes(dualvolumes, 1)
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
                          coords::IDVector{0,SVector{C,S}},
                          dualcoords::IDVector{R,SVector{C,S}},
                          dualcoords1::IDVector{R1,SVector{C,S}},
                          dualvolumes1::IDVector{R1,S},
                          signature::Int=1) where {dualkind,D,R,R1,C,S}
    R::Int
    R1::Int
    C::Int
    @assert 0 ≤ R ≤ R1 ≤ D
    @assert R1 == R + 1
    nvertices, nsimplices = size(simplices)
    dualvolumes = IDVector{R}(Array{S}(undef, nsimplices))
    # Loop over all `R`-simplices
    @inbounds for i in axes(dualvolumes, 1)
        si = sparse_column_rows(simplices, i)
        @assert length(si) == R + 1
        si = SVector{R + 1}(si[n] for n in 1:(R + 1))
        xsi = SVector{R + 1}(Form{C,1}(coords[i]) for i in si)
        bci = barycentre(xsi)::Form{C,1,S}
        cci = Form{C,1,S}(dualcoords[i])

        voli = zero(S)
        # Loop over all neighbouring `R+1`-simplices
        for j in sparse_column_rows(parents, i)
            sj = sparse_column_rows(simplices1, j)
            @assert length(sj) == R + 2
            sj = SVector{R + 2}(sj[n] for n in 1:(R + 2))

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
