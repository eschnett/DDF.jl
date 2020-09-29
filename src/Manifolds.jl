module Manifolds

using ComputedFieldTypes
using DifferentialForms
using ForwardDiff
using NearestNeighbors
using Optim: Optim, optimize
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

export DualKind, BarycentricDuals, CircumcentricDuals
@enum DualKind BarycentricDuals CircumcentricDuals

const dualkind = BarycentricDuals

# Weighted duals need to be circumcentric duals
const use_weighted_duals = false

################################################################################

const OpDict{K,T} = Dict{K,SparseOp{<:Any,<:Any,T}} where {K,T}

export Manifold
"""
A discrete manifold 
"""
struct Manifold{D,C,S}
    name::String

    # If `simplices[R][i,j]` is present, then vertex `i` is part of
    # the `R`-simplex `j`. `R ∈ 0:D`. We could omit `R=0`.
    simplices::OpDict{Int,One}
    # simplices::Dict{Int,Array{Int,2}}

    # The boundary ∂ of `0`-forms vanishes and is not stored. If
    # `boundaries[R][i,j] = s`, then `R`-simplex `j` has
    # `(R-1)`-simplex `i` as boundary with orientation `s`. `R ∈ 1:D`.
    boundaries::OpDict{Int,Int8}

    # Lookup tables from `Ri`-simplices to `Rj`-simplices. If
    # `lookup[(Ri,Rj)][i,j]` is present, then `Rj`-simplex `j`
    # contains `Ri`-simplex `i`. `Ri ∈ 0:D, Rj ∈ 0:D`. Many of these
    # are trivial and could be omitted.
    lookup::OpDict{Tuple{Int,Int},One}
    # lookup::Dict{Tuple{Int,Int},Array{Int,2}}

    # coords::Array{S,2}
    coords::Dict{Int,Vector{SVector{C,S}}}
    volumes::Dict{Int,Vector{S}}

    # Vertex weights [arXiv:math/0508188], [DOI:10.1145/2602143]
    weights::Vector{S}

    # Coordinates of vertices of dual grid, i.e.
    # barycentres/circumcentres of primal top-simplices
    # dualcoords::Array{S,2}
    dualcoords::Dict{Int,Vector{SVector{C,S}}}
    # `dualvolumes[R]` are the volumes of the simplices dual to the
    # primal `R`-simplices
    dualvolumes::Dict{Int,Vector{S}}

    # Nearest neighbour tree for simplex vertices
    simplex_tree::KDTree{SVector{C,S},Euclidean,S}

    function Manifold{D,C,S}(name::String, simplices::OpDict{Int,One},
                             boundaries::OpDict{Int,Int8},
                             lookup::OpDict{Tuple{Int,Int},One},
                             coords::Dict{Int,Vector{SVector{C,S}}},
                             volumes::Dict{Int,Vector{S}}, weights::Vector{S},
                             dualcoords::Dict{Int,Vector{SVector{C,S}}},
                             dualvolumes::Dict{Int,Vector{S}},
                             simplex_tree::KDTree{SVector{C,S},Euclidean,S}) where {D,
                                                                                    C,
                                                                                    S}
        D::Int
        @assert 0 <= D <= C
        mfd = new{D,C,S}(name, simplices, boundaries, lookup, coords, volumes,
                         weights, dualcoords, dualvolumes, simplex_tree)
        @assert invariant(mfd)
        return mfd
    end
    function Manifold(name::String, simplices::OpDict{Int,One},
                      boundaries::OpDict{Int,Int8},
                      lookup::OpDict{Tuple{Int,Int},One},
                      coords::Dict{Int,Vector{SVector{C,S}}},
                      volumes::Dict{Int,Vector{S}}, weights::Vector{S},
                      dualcoords::Dict{Int,Vector{SVector{C,S}}},
                      dualvolumes::Dict{Int,Vector{S}},
                      simplex_tree::KDTree{SVector{C,S},Euclidean,S}) where {C,
                                                                             S}
        D = maximum(keys(simplices))
        return Manifold{D,C,S}(name, simplices, boundaries, lookup, coords,
                               volumes, weights, dualcoords, dualvolumes,
                               simplex_tree)
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
    for R in 0:D
        println(io, "    coords[$R]=$(mfd.coords[R])")
    end
    for R in 0:D
        println(io, "    volumes[$R]=$(mfd.volumes[R])")
    end
    println(io, "    weights=$(mfd.weights)")
    for R in 0:D
        println(io, "    dualcoords[$R]=$(mfd.dualcoords[R])")
    end
    for R in 0:D
        println(io, "    dualvolumes[$R]=$(mfd.dualvolumes[R])")
    end
    return print(io, ")")
end

function Defs.invariant(mfd::Manifold{D,C})::Bool where {D,C}
    D >= 0 || (@assert false; return false)
    C >= D || (@assert false; return false)

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
    Set(keys(mfd.lookup)) == Set((Ri, Rj) for Ri in 0:D for Rj in 0:D) ||
        (@assert false; return false)
    for Ri in 0:D, Rj in 0:D
        lookup = mfd.lookup[(Ri, Rj)]::SparseOp{Ri,Rj,One}
        size(lookup) == (nsimplices(mfd, Ri), nsimplices(mfd, Rj)) ||
            (@assert false; return false)
        if Rj >= Ri
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
            mfd.lookup[(Ri, Rj)] == mfd.lookup[(Rj, Ri)]' ||
                (@assert false; return false)
        end
    end

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

    Set(keys(mfd.dualcoords)) == Set(0:D) || (@assert false; return false)
    for R in 0:D
        length(mfd.dualcoords[R]) == nsimplices(mfd, R) ||
            (@assert false; return false)
    end

    Set(keys(mfd.dualvolumes)) == Set(0:D) || (@assert false; return false)
    for R in 0:D
        length(mfd.dualvolumes[R]) == nsimplices(mfd, R) ||
            (@assert false; return false)
    end

    return true
end

# Comparison

function Base.:(==)(mfd1::Manifold{D,C}, mfd2::Manifold{D,C})::Bool where {D,C}
    mfd1 === mfd2 && return true
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
Return random point in manifold
"""
function random_point(::Val{R}, mfd::Manifold{D,C,S}) where {D,C,R,S}
    @assert 0 <= R <= D <= C
    C == 0 && return SVector{C,S}()
    N = R + 1
    # Choose simplex
    i = rand(1:nsimplices(mfd, R))
    si = sparse_column_rows(mfd.simplices[R], i)
    @assert length(si) == R + 1
    # Choose point in simplex
    λ = abs.(randn(SVector{C + 1,S}))
    λ /= norm(λ)
    x = sum(λ[n] * mfd.coords[0][si[n]] for n in 1:N)
    return x::SVector{C,S}
end

################################################################################

# Outer constructor

function Manifold(name::String, simplicesD::SparseOp{0,D,One},
                  coords0::Vector{SVector{C,S}},
                  weights::Vector{S}) where {D,C,S}
    @assert 0 <= D <= C

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

    # Calculate lookup tables
    lookup = OpDict{Tuple{Int,Int},One}()
    # Simplex definitions
    for R in 0:D
        if !haskey(lookup, (0, R))
            # println("$D (0,$R) simplices")
            lookup[(0, R)] = simplices[R]
        end
    end
    # Identity
    for R in 0:D
        nsimplicesR = size(simplices[R], 2)
        if !haskey(lookup, (R, R))
            # println("$D ($R,$R) id")
            lookup[(R, R)] = SparseOp{R,R}(sparse(1:nsimplicesR, 1:nsimplicesR,
                                                  fill(One(), nsimplicesR)))
        end
    end
    # Absolute value of boundaries
    for R in 1:D
        if !haskey(lookup, (R - 1, R))
            # println("$D ($(R-1),$R) boundaries")
            lookup[(R - 1, R)] = map(x -> One(x != 0), boundaries[R])
        end
    end
    # Chain of two lookup tables
    for Ri in 0:D, Rj in (Ri + 2):D
        if !haskey(lookup, (Ri, Rj))
            # println("$D ($Ri,$Rj) mul")
            Rk = Rj - 1
            lookup[(Ri, Rj)] = map(x -> One(x != 0),
                                   lookup[(Ri, Rk)] * lookup[(Rk, Rj)])
        end
    end
    # Transpose
    for Ri in 0:D, Rj in 0:(Ri - 1)
        @assert !haskey(lookup, (Ri, Rj))
        # println("$D ($Ri,$Rj) transpose")
        lookup[(Ri, Rj)] = lookup[(Rj, Ri)]'
    end
    # Check
    for Ri in 0:D, Rj in 0:D
        lookup[(Ri, Rj)]::SparseOp{Ri,Rj,One}
    end

    # Calculate coordinates and volumes
    coords = Dict{Int,Vector{SVector{C,S}}}()
    volumes = Dict{Int,Vector{S}}()
    for R in 0:D
        # TODO: Combine these two calculations
        coords[R] = calc_coords(simplices[R], coords0)
        volumes[R] = calc_volumes(simplices[R], coords0)
        @assert all(x -> x != 0 && isfinite(x), volumes[R])
    end

    if use_weighted_duals
        @assert dualkind == CircumcentricDuals
        # Optimize weights
        weights = optimize_weights(Val(dualkind), Val(D), simplices, lookup,
                                   coords[0], volumes, weights)

        if false
            # Optimize vertices
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
                # @assert all(s -> -1 <= s <= 1, boundary_faces)
                @assert all(s -> -2 <= s <= 2, boundary_faces)
                interior_vertices = ones(Bool, length(coords[0]))
                @assert length(boundary_faces) == size(simplices[D - 1], 2)
                for j in 1:size(simplices[D - 1], 2)
                    if isodd(boundary_faces[j])
                        for i in sparse_column_rows(simplices[D - 1], j)
                            interior_vertices[i] = false
                        end
                    end
                end
            else
                interior_vertices = zeros(Bool, length(coords[0]))
            end
            interior_vertices::Vector{Bool}
            # TODO: Allow moving boundary points tangential to the boundary
            optimize_vertices(Val(dualkind), Val(D), simplices, lookup,
                              coords[0], volumes, weights, interior_vertices)
        end
    end

    # Calculate dual coordinates and dual volumes
    dualcoords = Dict{Int,Vector{SVector{C,S}}}()
    dualvolumes = Dict{Int,Vector{S}}()
    # TODO: Combine these two calculations?
    if use_weighted_duals
        for R in 0:D
            dualcoords[R] = calc_dualcoords(Val(dualkind), simplices[R],
                                            coords[0], weights)
        end
    else
        for R in 0:D
            dualcoords[R] = calc_dualcoords(Val(dualkind), simplices[R],
                                            coords[0])
        end
    end
    if dualkind == BarycentricDuals
        for R in 0:D
            dualvolumes[R] = calc_dualvolumes(Val(dualkind), Val(D), Val(R),
                                              simplices[R], lookup, coords[0],
                                              volumes[D], dualcoords)
            @assert all(x -> x != 0 && isfinite(x), dualvolumes[R])
        end
    elseif dualkind == CircumcentricDuals
        dualvolumes[D] = fill(S(1), nsimplices)
        for R in (D - 1):-1:0
            dualvolumes[R] = calc_dualvolumes(Val(dualkind), Val(D), Val(R),
                                              simplices, lookup, coords[0],
                                              dualcoords)
        end
    else
        @assert false
    end

    # Ensure that all dual volumes are strictly positive
    allpositive = true
    for R in 0:D
        allpositive &= all(>(0), dualvolumes[R])
    end
    if !allpositive
        @warn "Not all dual volumes are strictly positive"
    end

    if D > 0
        check_delaunay(simplices[D], lookup[(D - 1, D)], lookup[(D, D - 1)],
                       coords[0], dualcoords[D])
    end

    simplex_tree = KDTree(coords[0])

    # Create D-manifold
    return Manifold(name, simplices, boundaries, lookup, coords, volumes,
                    weights, dualcoords, dualvolumes, simplex_tree)
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

"""
Optimize vertex weights
"""
function optimize_weights(::Val{dualkind}, ::Val{D}, simplices::OpDict{Int,One},
                          lookup::OpDict{Tuple{Int,Int},One},
                          coords::Vector{SVector{C,S}},
                          volumes::Dict{Int,Vector{S}},
                          weights::Vector{S}) where {dualkind,D,C,S}
    # @show "optimize_weights.0" D C
    D::Int
    C::Int
    @assert 0 <= D <= C

    D == 0 && return weights

    nvertices, nsimplices = size(simplices[D])
    @assert length(coords) == nvertices
    @assert length(weights) == nvertices

    dual_count = 0
    function calc_duals(weights::Vector{S}) where {S}
        dual_count += 1
        # @show "w calc_duals.0" D dual_count length(weights)
        local dualcoords = Dict{Int,Vector{SVector{C,S}}}()
        for R in 0:D
            dualcoords[R] = calc_dualcoords(Val(dualkind), simplices[R], coords,
                                            weights)
        end

        local dualvolumes = Dict{Int,Vector{S}}()
        dualvolumes[D] = fill(S(1), nsimplices)
        local cost = S(0)
        for R in (D - 1):-1:0
            dualvolumes[R], cost_R = calc_dualvolumes_cost(Val(dualkind),
                                                           Val(D), simplices[R],
                                                           simplices[R + 1],
                                                           lookup[(R + 1, R)],
                                                           coords, volumes[R],
                                                           dualcoords[R],
                                                           dualcoords[R + 1],
                                                           dualvolumes[R + 1])
            cost += cost_R
        end

        # @show "calc_duals.9" cost typeof(cost)
        if cost isa ForwardDiff.Dual
            # @show length(cost.partials)
        end
        return dualcoords, dualvolumes, cost
    end

    function calc_cost(weights)
        local dc, dv, c = calc_duals(weights)
        return c
    end
    dc1 = nothing
    dv1 = nothing
    function calc_cost1(weights)
        dc1, dv1, c = calc_duals(weights)
        return c
    end

    if nsimplices > 0
        # Newton's method requires the Hessian and is slow.
        # fun = Optim.TwiceDifferentiable(calc_cost, weights;
        #                                autodiff = :forward)
        # result = optimize(fun, weights, Optim.Options(iterations = 1000))
        # Forward-mode automatic differencing is slow
        # fun = Optim.OnceDifferentiable(calc_cost, weights;
        #                                autodiff = :forward)
        # result = optimize(fun, weights, Optim.BFGS(),
        #                   Optim.Options(iterations = 1000))
        fun = calc_cost
        result = optimize(fun, weights, Optim.NelderMead(),
                          Optim.Options(iterations=1000))
        println(result)
        weights = result.minimizer
    end
    weights::Vector{S}

    # Calculate dual coordinates, dual volumes, and cost function
    dualcoords, dualvolumes, cost = calc_duals(weights)
    # dcost = ForwardDiff.gradient(calc_cost1, weights)
    ddualcoords, ddualvolumes = dc1, dv1

    # Ensure that all dual volumes are strictly positive
    allpositive = true
    num_nonpositive = 0
    num_toosmall = 0
    min_scaledvol = S(Inf)
    for R in 0:D
        if !isempty(dualvolumes[R])
            volR = sum(dualvolumes[R]) / length(dualvolumes[R])
            allpositive &= all(>=(volR / 100), dualvolumes[R])
            num_nonpositive += count(<=(0), dualvolumes[R])
            num_toosmall += count(<(volR / 100), dualvolumes[R])
            min_scaledvol = min(min_scaledvol, minimum(dualvolumes[R]) / volR)
        end
    end

    println("optimize_weights: Dual volumes: nonpositive: $num_nonpositive, ",
            "toosmall: $num_toosmall, minscaled: $min_scaledvol, cost: $cost")
    # println("weights: ", weights)
    # for R in 0:D
    #     println("dualcoords[$R]: ", dualcoords[R])
    #     println("dualvolumes[$R]: ", dualvolumes[R])
    #     println("d(dualcoords)/d(weights)[$R]: ",
    #             map(x -> map(y -> y.partials, x), ddualcoords[R]))
    #     println("d(dualvolumes)/d(weights)[$R]: ",
    #             map(x -> map(y -> y.partials, x), ddualvolumes[R]))
    # end
    # println("d(cost)/d(weights): $dcost")

    @assert allpositive

    # @show "optimize_weights.9" D C
    return weights
end

"""
Optimize vertex positions
"""
function optimize_vertices(::Val{dualkind}, ::Val{D},
                           simplices::OpDict{Int,One},
                           lookup::OpDict{Tuple{Int,Int},One},
                           coords::Vector{SVector{C,S}},
                           volumes::Dict{Int,Vector{S}}, weights::Vector{S},
                           interior_vertices::Vector{Bool}) where {dualkind,D,C,
                                                                   S}
    # @show "optimize_vertices.0" D C
    D::Int
    C::Int
    @assert 0 <= D <= C

    D == 0 && return coords

    nvertices, nsimplices = size(simplices[D])
    @assert length(coords) == nvertices
    @assert length(weights) == nvertices
    @assert length(interior_vertices) == nvertices

    function extract_coords()
        # # @show "extract_coords.0"
        local coords′ = Vector{S}(undef, C * count(interior_vertices))
        j = 0
        for i in 1:length(coords)
            if interior_vertices[i]
                for c in 1:C
                    coords′[j + c] = coords[i][c]
                end
                j += C
            end
        end
        @assert j == length(coords′)
        # # @show "extract_coords.9"
        return coords′
    end

    function insert_coords(coords′::Vector{S′}) where {S′}
        # # @show "insert_coords.0"
        @assert length(coords′) == C * count(interior_vertices)
        local coords1 = Vector{SVector{C,S′}}(undef, length(coords))
        j = 0
        for i in 1:length(coords1)
            if interior_vertices[i]
                coords1[i] = SVector{C,S′}(@view coords′[(j + 1):(j + C)])
                j += C
            else
                coords1[i] = SVector{C,S′}(coords[i]::SVector{C,S})
            end
        end
        @assert j == length(coords′)
        # # @show "insert_coords.9"
        return coords1
    end

    dual_count = 0
    function calc_duals(coords′::Vector{S}) where {S}
        dual_count += 1
        # @show "v calc_duals.0" dual_count
        local coords0 = insert_coords(coords′)

        # Calculate coordinates and volumes
        local coords = Dict{Int,Vector{SVector{C,S}}}()
        local volumes = Dict{Int,Vector{S}}()
        for R in 0:D
            # TODO: Combine these two calculations
            coords[R] = calc_coords(simplices[R], coords0)
            volumes[R] = calc_volumes(simplices[R], coords0)
        end

        local dualcoords = Dict{Int,Vector{SVector{C,S}}}()
        for R in 0:D
            dualcoords[R] = calc_dualcoords(Val(dualkind), simplices[R],
                                            coords0, weights)
        end

        local dualvolumes = Dict{Int,Vector{S}}()
        dualvolumes[D] = fill(S(1), nsimplices)
        local cost = S(0)
        for R in (D - 1):-1:0
            dualvolumes[R], cost_R = calc_dualvolumes_cost(Val(dualkind),
                                                           Val(D), simplices[R],
                                                           simplices[R + 1],
                                                           lookup[(R + 1, R)],
                                                           coords0, volumes[R],
                                                           dualcoords[R],
                                                           dualcoords[R + 1],
                                                           dualvolumes[R + 1])
            cost += cost_R
        end

        # @show "calc_duals.9"
        return dualcoords, dualvolumes, cost
    end

    function calc_cost(coords′)
        local dc, dv, c = calc_duals(coords′)
        return c
    end
    dc1 = nothing
    dv1 = nothing
    function calc_cost1(coords′)
        dc1, dv1, c = calc_duals(coords′)
        return c
    end

    coords′ = extract_coords()
    if !isempty(coords′)
        if length(coords′) == 1
            fun = Optim.OnceDifferentiable(calc_cost, coords′;
                                           autodiff=:forward)
            result = optimize(fun, coords′, Optim.Options(iterations=1000))
        else
            # Newton's method requires the Hessian and is slow.
            # fun = Optim.TwiceDifferentiable(calc_cost, coords′;
            #                                 autodiff = :forward)
            # result = optimize(fun, coords′, Optim.Options(iterations = 1000))
            # Forward-mode automatic differencing is slow
            # fun = Optim.OnceDifferentiable(calc_cost, coords′;
            #                                autodiff = :forward)
            # result = optimize(fun, coords′, Optim.Options(iterations = 1000))
            fun = calc_cost
            result = optimize(fun, coords′, Optim.NelderMead(),
                              Optim.Options(
                                            # iterations = 1000
                                            iterations=100))
        end
        println(result)
        coords′ = result.minimizer
    end
    coords′::Vector{S}
    coords = insert_coords(coords′)
    coords::Vector{SVector{C,S}}

    # Calculate dual coordinates, dual volumes, and cost function
    dualcoords, dualvolumes, cost = calc_duals(coords′)
    # dcost = ForwardDiff.gradient(calc_cost1, coords′)
    ddualcoords, ddualvolumes = dc1, dv1

    # Ensure that all dual volumes are strictly positive
    allpositive = true
    num_nonpositive = 0
    num_toosmall = 0
    min_scaledvol = S(Inf)
    for R in 0:D
        if !isempty(dualvolumes[R])
            volR = sum(dualvolumes[R]) / length(dualvolumes[R])
            allpositive &= all(>=(volR / 100), dualvolumes[R])
            num_nonpositive += count(<=(0), dualvolumes[R])
            num_toosmall += count(<(volR / 100), dualvolumes[R])
            min_scaledvol = min(min_scaledvol, minimum(dualvolumes[R]) / volR)
        end
    end

    println("optimize_vertices: Dual volumes: nonpositive: $num_nonpositive, ",
            "toosmall: $num_toosmall, minscaled: $min_scaledvol, cost: $cost")
    # println("weights: ", weights)
    # for R in 0:D
    #     println("dualcoords[$R]: ", dualcoords[R])
    #     println("dualvolumes[$R]: ", dualvolumes[R])
    #     println("d(dualcoords)/d(weights)[$R]: ",
    #             map(x -> map(y -> y.partials, x), ddualcoords[R]))
    #     println("d(dualvolumes)/d(weights)[$R]: ",
    #             map(x -> map(y -> y.partials, x), ddualvolumes[R]))
    # end
    # println("d(cost)/d(weights): $dcost")

    @assert allpositive

    # @show "optimize_vertices.9" D C
    return coords
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
Calculate weighted circumcentric dual coordinates
"""
function calc_dualcoords(::Val{CircumcentricDuals},
                         simplices::SparseOp{0,D,One},
                         coords::Vector{SVector{C,Sc}},
                         weights::AbstractVector{Sw}) where {D,C,Sc,Sw}
    nvertices, nsimplices = size(simplices)
    @assert length(coords) == nvertices
    @assert length(weights) == nvertices
    D == 0 && return coords
    Sd = typeof(one(Sw) * zero(Sc))
    dualcoords = Array{SVector{C,Sd}}(undef, nsimplices)
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
    @assert 0 <= R <= D <= C

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
                          coords::Vector{SVector{C,Sc}}, volumes::Vector{Sc},
                          dualcoords::Vector{SVector{C,Sd}},
                          dualcoords1::Vector{SVector{C,Sd}},
                          dualvolumes1::Vector{Sd}) where {dualkind,D,R,R1,C,Sc,
                                                           Sd}
    dualvolumes, cost = calc_dualvolumes_cost(Val(CircumcentricDuals), Val(D),
                                              simplices, simplices1, parents,
                                              coords, volumes, dualcoords,
                                              dualcoords1, dualvolumes1)
    return dualvolumes
end

function calc_dualvolumes_cost(::Val{CircumcentricDuals}, ::Val{D},
                               simplices::SparseOp{0,R,One},
                               simplices1::SparseOp{0,R1,One},
                               parents::SparseOp{R1,R,One},
                               coords::Vector{SVector{C,Sc}},
                               volumes::Vector{Sc},
                               dualcoords::Vector{SVector{C,Sd}},
                               dualcoords1::Vector{SVector{C,Sd}},
                               dualvolumes1::Vector{Sd}) where {D,R,R1,C,Sc,Sd}
    R::Int
    R1::Int
    C::Int
    S = valtype1(Sc)
    @assert !(S <: ForwardDiff.Dual)
    @assert 0 <= R <= R1 <= D
    @assert R1 == R + 1
    nvertices, nsimplices = size(simplices)
    dualvolumes = Array{Sd}(undef, nsimplices)
    cost = zero(Sd)
    # Loop over all `R`-simplices
    @inbounds for i in 1:nsimplices
        si = sparse_column_rows(simplices, i)
        @assert length(si) == R + 1
        xsi = SVector{R + 1}(Form{C,1}(coords[i]) for i in si)
        bci = barycentre(xsi)::Form{C,1,Sc}
        cci = Form{C,1,Sd}(dualcoords[i])

        voli = zero(Sd)
        costi = zero(Sd)
        # This line is expensive -- it shouldn't be
        cost_size = volumes[i]^(S(1) / R)
        # Loop over all neighbouring `R+1`-simplices
        for j in sparse_column_rows(parents, i)
            sj = sparse_column_rows(simplices1, j)
            @assert length(sj) == R + 2

            xsj = SVector{R + 2}(Form{C,1}(coords[j]) for j in sj)
            bcj = barycentre(xsj)::Form{C,1,Sc}
            ccj = Form{C,1,Sd}(dualcoords1[j])

            ysi = map(y -> y - xsi[1], deleteat(xsi, 1))   # R
            ysj = map(y -> y - xsj[1], deleteat(xsj, 1))   # R+1
            @assert !isempty(ysj)
            ni = ∧(ysj) ⋅ ∧(ysi)   # 1
            ni::Form{D,1,Sc}
            qsi = map(y -> norm(ni ⋅ y) < 10 * eps(S), ysi)
            @assert all(qsi)
            lni = norm(ni)
            @assert lni > 0
            nni = ni / lni

            s0 = (bcj - xsi[1]) ⋅ nni
            s0::Form{D,0,Sc}
            s = bitsign(signbit(s0[]))
            h0 = (ccj - cci) ⋅ nni
            h0::Form{D,0,Sd}
            h = s * h0[]

            b = dualvolumes1[j]
            volj = b * h
            costj = (h / cost_size - S(1) / D)^2
            # costj = volumes[i] * norm2(bcj - ccj)

            voli += volj
            costi += costj
        end

        dualvolumes[i] = voli / (D - R)

        cost += costi
    end
    return dualvolumes, cost
end

valtype1(::Type{T}) where {T} = T
valtype1(::Type{ForwardDiff.Dual{T,V,N}}) where {T,V,N} = V

"""
Check Delaunay condition: No vertex must lie in the circumcentre of a
simplex
"""
function check_delaunay(simplices::SparseOp{0,D,One},
                        lookup::SparseOp{D1,D,One}, lookup1::SparseOp{D,D1,One},
                        coords::Vector{SVector{C,S}},
                        dualcoords::Vector{SVector{C,S}}) where {D,D1,C,S}
    D::Int
    D1::Int
    @assert 0 <= D1 <= D
    @assert D1 == D - 1
    C::Int

    # This is currently broken because the location of the dual
    # coordinates are not the circumcentres any more. We need to
    # recalculate the circumcentres.
    return

    for i in 1:size(simplices, 2)
        si = sparse_column_rows(simplices, i)
        @assert length(si) == D + 1
        x1i = Form{C,1}(coords[first(si)])
        cci = Form{C,1}(dualcoords[i])
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
                            xl = Form{C,1}(coords[l])
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
