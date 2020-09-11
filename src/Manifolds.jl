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

# Note: BarycentricDuals are not yet implemented
const dualkind = CircumcentricDuals

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

# TODO: This is unused???
export ZeroVolumeException
struct ZeroVolumeException <: Exception
    D::Int
    C::Int
    i::Int
    simplex::Vector{Int}
    cs::Vector                  # Vector{SVector{C,T}}
end

struct Face{N}
    vertices::SVector{N,Int}
    parent::Int
    parity::Int8
end

function Manifold(name::String, simplices::SparseOp{0,D,One},
                  coords::Vector{SVector{C,S}},
                  weights::Vector{S}) where {D,C,S}
    @assert 0 <= D <= C

    nvertices, nsimplices = size(simplices)
    @assert length(coords) == nvertices

    if D == 0
        volumes = fill(S(1), nsimplices)
        dualcoords = coords
        dualvolumes = fill(S(1), nvertices)
        # lookup = SparseOp{D,D}(sparse(1:nsimplices, 1:nsimplices,
        #                               fill(One(), nsimplices)))
        lookup = simplices
        simplex_tree = KDTree(coords)
        return Manifold(name, OpDict{Int,One}(0 => simplices),
                        OpDict{Int,Int8}(),
                        OpDict{Tuple{Int,Int},One}((0, 0) => lookup),
                        Dict{Int,Vector{SVector{C,S}}}(0 => coords),
                        Dict{Int,Vector{S}}(0 => volumes), weights,
                        Dict{Int,Vector{SVector{C,S}}}(0 => dualcoords),
                        Dict{Int,Vector{S}}(0 => dualvolumes), simplex_tree)
    end

    # Calculate lower-dimensional simplices
    # See arXiv:1103.3076v2 [cs.NA], section 7
    facelist = Face{D}[]
    for j in 1:size(simplices, 2)
        ks = SVector{D + 1,Int}(sparse_column_rows(simplices, j)...)
        for n in 1:(D + 1)
            # Leave out vertex n
            ls = deleteat(ks, n)
            p = bitsign(isodd(n - 1))
            push!(facelist, Face{D}(ls, j, p))
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

    # Recursively create lower-dimensional (D-1)-manifold
    mfd1 = Manifold(name, faces, coords, weights)

    newsimplices = mfd1.simplices
    newsimplices[D] = simplices
    newboundaries = mfd1.boundaries
    newboundaries[D] = boundaries

    # Extend lookup table
    newlookup = mfd1.lookup
    newlookup[(0, D)] = simplices
    newlookup[(D, D)] = SparseOp{D,D}(sparse(1:nsimplices, 1:nsimplices,
                                             fill(One(), nsimplices)))
    if D > 1
        newlookup[(D - 1, D)] = map(x -> One(x != 0), boundaries)
        for Ri in 1:(D - 2)
            newlookup[(Ri, D)] = map(x -> One(x != 0),
                                     newlookup[(Ri, D - 1)] *
                                     newlookup[(D - 1, D)])
        end
    end
    for Rj in 0:(D - 1)
        newlookup[(D, Rj)] = newlookup[(Rj, D)]'
    end
    for Ri in 0:D, Rj in 0:D
        newlookup[(Ri, Rj)]::SparseOp{Ri,Rj,One}
    end

    # Calculate coordinates and volumes
    newcoords = mfd1.coords
    newcoords[D] = calc_coords(simplices, coords)
    newvolumes = mfd1.volumes
    newvolumes[D] = calc_volumes(simplices, coords)

    # Optimize weights

    # Calculate dual coordinates and dual volumes
    newdualcoords = Dict{Int,Vector{SVector{C,S}}}()
    newdualvolumes = Dict{Int,Vector{S}}()
    if C == D
        weights, newdualcoords, newdualvolumes = optimize_weights(Val(dualkind),
                                                                  Val(D),
                                                                  newsimplices,
                                                                  newlookup,
                                                                  coords,
                                                                  newvolumes,
                                                                  weights)

        # Ensure that all dual volumes are strictly positive
        allpositive = true
        for R in 0:D
            allpositive &= all(>(0), newdualvolumes[R])
        end
        if !allpositive
            @show "!allpositive"
            @show D
            @show weights
            @show newdualcoords
            @show newdualvolumes
        end
        @assert allpositive
    else
        # dummy data, won't be used
        for R in 0:D
            newdualcoords[R] = zeros(SVector{C,S}, size(newsimplices[R], 2))
            newdualvolumes[R] = zeros(S, size(newsimplices[R], 2))
        end
    end

    # Only test for the final manifold
    if C == D
        check_delaunay(simplices, newlookup[(D - 1, D)], newlookup[(D, D - 1)],
                       coords, newdualcoords[D])
    end

    # Create D-manifold
    return Manifold(name, newsimplices, newboundaries, newlookup, newcoords,
                    newvolumes, weights, newdualcoords, newdualvolumes,
                    mfd1.simplex_tree)
end

################################################################################

"""
Optimize vertex weights
"""
function optimize_weights(::Val{dualkind}, ::Val{D}, simplices::OpDict{Int,One},
                          lookup::OpDict{Tuple{Int,Int},One},
                          coords::Vector{SVector{C,S}},
                          volumes::Dict{Int,Vector{S}},
                          weights::Vector{S}) where {dualkind,D,C,S}
    D::Int
    C::Int
    @assert 0 <= D <= C

    nvertices, nsimplices = size(simplices[D])
    @assert length(coords) == nvertices
    @assert length(weights) == nvertices

    function calc_duals(weights::AbstractVector{S}) where {S}
        local dualcoords = Dict{Int,Vector{SVector{C,S}}}()
        for R in 0:D
            dualcoords[R] = calc_dualcoords(Val(dualkind), simplices[R], coords,
                                            weights)
        end

        local dualvolumes = Dict{Int,Vector{S}}()
        dualvolumes[D] = fill(S(1), nsimplices)
        local cost = S(0)
        for R in (D - 1):-1:0
            dualvolumes[R], cost_R = calc_dualvolumes(Val(dualkind), Val(D),
                                                      simplices[R],
                                                      simplices[R + 1],
                                                      lookup[(R + 1, R)],
                                                      coords, volumes[R],
                                                      dualcoords[R],
                                                      dualcoords[R + 1],
                                                      dualvolumes[R + 1])
            cost += cost_R
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

    oldweights = weights
    if nsimplices > 0
        result = optimize(calc_cost, weights; # , Optim.LBFGS();
                          autodiff = :forward, iterations = 1000)
        println(result)
        weights = result.minimizer
    else
        weights = copy(weights)
    end
    weights::Vector{S}

    # Calculate dual coordinates, dual volumes, and cost function
    dualcoords, dualvolumes, cost = calc_duals(weights)
    dcost = ForwardDiff.gradient(calc_cost1, weights)
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

    println("Dual volumes: nonpositive: $num_nonpositive, ",
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

    return weights, dualcoords, dualvolumes
end

################################################################################

"""
Calculate coordinates
"""
function calc_coords(simplices::SparseOp{0,D,One},
                     coords0::Vector{SVector{C,S}}) where {D,C,S}
    nvertices, nsimplices = size(simplices)
    coords = Array{SVector{C,S}}(undef, nsimplices)
    for i in 1:nsimplices
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
    volumes = Array{S}(undef, nsimplices)
    for i in 1:nsimplices
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
    nvertices, nsimplices = size(simplices)
    dualcoords = Array{SVector{C,S}}(undef, nsimplices)
    for i in 1:nsimplices
        si = SVector{D + 1,Int}(sparse_column_rows(simplices, i)...)
        xs = SVector{D + 1,SVector{C,S}}(coords[i] for i in si)
        dualcoords[i] = barycentre(xs)
    end
    return dualcoords
end

"""
Calculate weighted circumcentric dual coordinates
"""
function calc_dualcoords(::Val{CircumcentricDuals},
                         simplices::SparseOp{0,D,One},
                         coords::Vector{SVector{C,S}},
                         weights::AbstractVector{S′}) where {D,C,S,S′}
    nvertices, nsimplices = size(simplices)
    @assert length(coords) == nvertices
    @assert length(weights) == nvertices
    dualcoords = Array{SVector{C,S′}}(undef, nsimplices)
    for i in 1:nsimplices
        si = SVector{D + 1,Int}(sparse_column_rows(simplices, i)...)
        xs = SVector{D + 1}(Form{C,1}(coords[i]) for i in si)
        ws = SVector{D + 1}(Form{C,0}((weights[i],)) for i in si)
        dualcoords[i] = circumcentre(xs, ws)
    end
    return dualcoords
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
                          dualcoords::Vector{SVector{C,S′}},
                          dualcoords1::Vector{SVector{C,S′}},
                          dualvolumes1::Vector{S′}) where {D,R,R1,C,S,S′}
    R::Int
    R1::Int
    C::Int
    @assert 0 <= R <= R1 <= D
    @assert R1 == R + 1
    nvertices, nsimplices = size(simplices)
    dualvolumes = Array{S′}(undef, nsimplices)
    cost = zero(S′)
    # Loop over all `R`-simplices
    for i in 1:nsimplices
        si = sparse_column_rows(simplices, i)
        @assert length(si) == R + 1
        xsi = SVector{R + 1}(Form{C,1}(coords[i]) for i in si)
        bci = barycentre(xsi)::Form{C,1,S}
        cci = Form{C,1,S′}(dualcoords[i])

        voli = zero(S′)
        costi = zero(S′)
        cost_size = volumes[i]^(S(1) / R)
        # Loop over all neighbouring `R+1`-simplices
        for j in sparse_column_rows(parents, i)
            sj = sparse_column_rows(simplices1, j)
            @assert length(sj) == R + 2

            xsj = SVector{R + 2}(Form{C,1}(coords[j]) for j in sj)
            bcj = barycentre(xsj)::Form{C,1,S}
            ccj = Form{C,1,S′}(dualcoords1[j])

            ysi = map(y -> y - xsi[1], deleteat(xsi, 1))   # R
            ysj = map(y -> y - xsj[1], deleteat(xsj, 1))   # R+1
            @assert !isempty(ysj)
            ni1 = ∧(Tuple(ysj)...)
            ni2 = isempty(ysi) ? one(xsi[1]) : ∧(Tuple(ysi)...)
            ni = ni1 ⋅ ni2   # 1
            ni::Form{D,1,S}
            qsi = map(y -> norm(ni ⋅ y) < 10 * eps(S), ysi)
            @assert all(qsi)
            lni = norm(ni)
            @assert lni > 0
            nni = ni / lni

            s0 = (bcj - xsi[1]) ⋅ nni
            s0::Form{D,0,S}
            s = bitsign(signbit(s0[]))
            h0 = (ccj - cci) ⋅ nni
            h0::Form{D,0,S′}
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
