#TODO: Rename to "metric spaces"
module Geometries

using Bernstein
using ComputedFieldTypes
using Delaunay
using DifferentialForms
using Distances
using LinearAlgebra
using NearestNeighbors
using OrderedCollections
using SimplexQuad
using SparseArrays
using StaticArrays

using ..Algorithms
using ..Defs
using ..Funs
using ..Ops
using ..Topologies

# We use the following conventions:
#     D::Int               number of dimension
#     S::Signature         signature of manifold
#     V::SubTopology       manifold
#     B::DirectSum.Basis   basis
#     R::Int               rank (of multivector)
# These can have a prefix:
#     no prefix   euclidean basis (for manifold)
#     H           homogeneous basis
#     C           conformal basis
#     (A          abstract (e.g. numbering tetrad vectors))

# export Domain
# @computed struct Domain{D, T}
#     xmin::DVector(D, T)
#     xmax::DVector(D, T)
#     function Domain{D,T}(xmin::Chain{V,1,T}, xmax::Chain{V,1,T}) where {D,V,T}
#         V::SubTopology
#         T::Type
#         @assert D == ndims(V)
#         new{D,T}(xmin, xmax)
#     end
#     Domain(xmin::Chain{V,1,T}, xmax::Chain{V,1,T}) where {V,T} =
#         Domain{ndims(V),T}(xmin, xmax)
# end
# 
# Defs.invariant(dom::Domain) = true

export ZeroVolumeException
struct ZeroVolumeException <: Exception
    D::Int
    R::Int
    i::Int
    simplex::Simplex            # Simplex{N,Int}
    cs::AbstractVector          # Vector{Form{D,1,T}}
end

export Geometry
@computed struct Geometry{D,T}
    name::String
    topo::Topology{D}
    # Coordinates of vertices
    coords::Fun{D,Pr,0,fulltype(Form{D,1,T})}
    # Volumes of 0-forms are always 1 and don't need to be stored
    volumes::Dict{Int,Fun{D,Pr,R,T} where R}
    # Coordinates of vertices of dual grid, i.e. circumcentres of top
    # simplices
    dualcoords::Fun{D,Dl,D,fulltype(Form{D,1,T})}
    # Dual volumes of dual top-forms are always 1 and don't need to be stored
    # dualvolumes[R = D-DR]
    dualvolumes::Dict{Int,Fun{D,Dl,R,T} where R}
    # Nearest neighbour tree for simplex vertices
    simplex_tree::KDTree{SVector{D,Float64},Euclidean,T}
    # TODO: dualsimplex_tree

    function Geometry{D,T}(name::String, topo::Topology{D},
                           coords::Fun{D,Pr,0,<:Form{D,1,T}},
                           volumes::Dict{Int,Fun{D,Pr,R,T} where R},
                           dualcoords::Fun{D,Dl,D,<:Form{D,1,T}},
                           dualvolumes::Dict{Int,Fun{D,Dl,R,T} where R},
                           simplex_tree::KDTree{SVector{D,Float64},Euclidean,T}) where {D,
                                                                                        T}
        D::Int
        T::Type
        @assert D >= 0
        @assert isempty(symdiff(keys(volumes), 0:D))
        @assert isempty(symdiff(keys(dualvolumes), 0:D))
        geom = new(name, topo, coords, volumes, dualcoords, dualvolumes,
                   simplex_tree)
        @assert invariant(geom)
        return geom
    end
end

function Base.show(io::IO, geom::Geometry{D,T}) where {D,T}
    println(io)
    println(io, "Geometry{$D,$T}(")
    println(io, "    name=$(geom.name)")
    println(io, "    topo=$(geom.topo)")
    println(io, "    coords=$(geom.coords)")
    for (d, vs) in sort!(OrderedDict(geom.volumes))
        println(io, "    volumes[$d]=$vs")
    end
    println(io, "    dualcoords=$(geom.dualcoords)")
    for (d, dvs) in sort!(OrderedDict(geom.dualvolumes))
        println(io, "    dualvolumes[$d]=$dvs")
    end
    return print(io, ")")
end

function Defs.invariant(geom::Geometry{D})::Bool where {D}
    invariant(geom.topo) || return false
    isempty(symdiff(keys(geom.volumes), 0:D)) || return false
    isempty(symdiff(keys(geom.dualvolumes), 0:D)) || return false
    return true
end

function Geometry(name::String, topo::Topology{D},
                  coords::Fun{D,Pr,0,<:Form{D,1,T}}) where {D,T}
    D::Int
    T::Type

    # Calculate volumes
    volumes = Dict{Int,Fun{D,Pr,R,T} where {R}}()
    for R in 0:D
        values = Array{T}(undef, size(R, topo))
        for (i, s) in enumerate(topo.simplices[R])
            cs = coords.values[s.vertices]
            xs = SVector{R,fulltype(Form{D,1,T})}(cs[n + 1] - cs[1]
                                                  for n in 1:R)
            if length(xs) == 0
                vol = one(T)
            else
                vol = abs(∧(xs...))
            end
            vol /= factorial(R)
            # TODO: Why not?
            # vol *= bitsign(s.signbit)
            vol > 0 || throw(ZeroVolumeException(D, R, i, s, cs))
            values[i] = vol
        end
        vols = Fun{D,Pr,R,T}(topo, values)
        volumes[R] = vols
    end

    # Calculate circumcentres
    values = Array{fulltype(Form{D,1,T})}(undef, size(D, topo))
    let R = D
        for (i, s) in enumerate(topo.simplices[R])
            cs = coords.values[s.vertices]
            cc = circumcentre(cs)
            values[i] = cc
        end
    end
    dualcoords = Fun{D,Dl,D,fulltype(Form{D,1,T})}(topo, values)

    # Check Delaunay condition:
    # No vertex must lie in the circumcentre of a simplex
    for (i, si) in enumerate(topo.simplices[D])
        xi1 = coords[si.vertices[1]]
        cc = dualcoords.values[i]
        cr2 = abs2(xi1 - cc)
        # TODO: Check only vertices of neighbouring simplices
        for (j, sj) in enumerate(topo.simplices[0])
            if j ∉ si.vertices
                xj = coords[sj.vertices[1]]
                d2 = abs2(xj - cc)
                @assert d2 >= cr2 || d2 ≈ cr2
            end
        end
    end

    # Check one-sidedness for boundary simplices:
    # TODO

    # # Check that all circumcentres lie inside their simplices
    # for (i, si) in enumerate(topo.simplices[D])
    #     xs = coords[si.vertices]
    #     cc = dualcoords.values[i]
    #     λ = cartesian2barycentric(
    #         map(x -> convert(SVector, x), xs),
    #         convert(SVector, cc),
    #     )
    #     @assert all(0 <= λi <= 1 for λi in λ)
    # end

    # Calculate circumcentric dual volumes
    # [1198555.1198667, page 5]
    dualvolumes = Dict{Int,Fun{D,Dl,R,T} where {R}}()
    for R in D:-1:0
        if R == D
            values = ones(T, size(R, topo))
        else
            bnds = topo.boundaries[R + 1]
            values = zeros(T, size(R, topo))
            sis = topo.simplices[R]::Vector{Simplex{R + 1,Int}}
            sjs = topo.simplices[R + 1]::Vector{Simplex{R + 2,Int}}
            for (i, si) in enumerate(sis)
                # TODO: This is expensive
                js = findnz(bnds[i, :])[1]
                for j in js
                    sj = sjs[j]
                    b = dualvolumes[R + 1][j]
                    # TODO: Calculate lower-rank circumcentres as
                    # intersection between boundary and the line
                    # connecting two simplices?
                    # TODO: Cache circumcentres ahead of time
                    @assert length(si.vertices) == R + 1
                    @assert length(sj.vertices) == R + 2
                    xsi = coords[si.vertices]
                    cci = circumcentre(xsi)
                    xsj = coords[sj.vertices]
                    ccj = circumcentre(xsj)
                    # TODO: Handle case where the volume should be
                    # negative (i.e. when the volume circumcentre ccj
                    # is on the "other" side of the face circumcentre
                    # cci) (Is the previous statement correct?)
                    h = abs(cci - ccj)
                    values[i] += b * h / factorial(D - R)
                end
            end
        end
        # @assert all(>(0), values)
        vols = Fun{D,Dl,R,T}(topo, values)
        dualvolumes[R] = vols
    end

    # # Calculate barycentres
    # values = Array{fulltype(Form{D,1,T})}(undef, size(D, topo))
    # let R = D
    #     for (i,s) in enumerate(topo.simplices[R])
    #         cs = sarray(fulltype(Form{D,1,T}),
    #                     n -> coords.values[s.vertices[n]],
    #                     Val(R+1))
    #         cc = +(cs...) / length(cs)
    #         values[i] = cc
    #     end
    # end
    # dualcoords = Fun{D, Dl, D, fulltype(Form{D,1,T})}(topo, values)
    #
    # # Calculate barycentric dual volumes
    # dualvolumes = Dict{Int, Fun{D,Dl,R,T} where {R}}()
    # for DR in 0:D
    #     R = D - DR
    #     if R == D
    #         values = ones(T, size(R, topo))
    #     else
    #         sjs = topo.simplices[R+1]::Vector{Simplex{R+2, Int}}
    #         bnds = topo.boundaries[R+1]
    #         values = zeros(T, size(R, topo))
    #         # Loop over all duals of rank R (e.g. faces)
    #         if R == 0
    #             sis = (Simplex{1,Int}(SVector(i)) for i in 1:topo.nvertices)
    #         else
    #             sis = topo.simplices[R]::Vector{Simplex{R+1, Int}}
    #         end
    #         for (i,si) in enumerate(sis)
    #             # Loop over all neighbours of i (e.g. volumes)
    #             # TODO: This is expensive
    #             js = findnz(bnds[i,:])[1]
    #             for j in js
    #                 sj = sjs[j]
    #                 si.vertices::SVector{R+1, Int}
    #                 sj.vertices::SVector{R+2, Int}
    #                 xsi = coords[si.vertices]
    #                 xsj = coords[sj.vertices]
    #                 ccj = +(xsj...) / length(xsj)
    #                 ysi = xsi .- ccj
    #                 vol = abs(∧(ysi...))
    #                 # TODO: take sign into account?
    #                 @assert vol > 0
    #                 values[i] += vol
    #             end
    #         end
    #     end
    #     @assert all(>(0), values)
    #     vols = Fun{D, Dl, R, T}(topo, values)
    #     dualvolumes[R] = vols
    # end

    simplex_tree = KDTree(map(x -> convert(SVector, x), coords.values))

    return Geometry{D,T}(name, topo, coords, volumes, dualcoords, dualvolumes,
                         simplex_tree)
end

# export bisect
# # Bisect simplex, creating a new geometry
# function bisect(geom::Geometry{D,T}) where {D,T}
#     # Special case -- no simplices
#     size(D, geom.topo) == 0 && return geom
#     # Ensure (for now) there is only one simplex
#     @assert size(D, geom.topo) == 1
#     @assert size(0, geom.topo) == D + 1
#     for R = 0:D
#         @assert size(R, geom.topo) == binomial(D + 1, R + 1)
#     end
#     # Choose new vertices
#     num_new_vertices = size(1, geom.topo)
#     new_vertices = size(0, geom.topo) .+ (1:num_new_vertices)
#     new_coords = [
#         (geom.coords.values[i] + geom.coords.values[j]) / 2
#         for i = 1:size(0, geom.topo), j = (i+1):size(0, geom.topo)
#     ]
#     @warn "move part of this to Topologies?"
#     return @error "continue here"
# end

export delaunay
function delaunay(name::String,
                  coords::AbstractVector{Form{D,1,T,X}})::Geometry{D,
                                                                   T} where {D,
                                                                             T,
                                                                             X}
    N = D + 1

    nvertices = length(coords)
    # Convert coordinates to 2d array
    xs = Array{T}(undef, nvertices, D)
    for i in 1:nvertices
        for d in 1:D
            xs[i, d] = coords[i][d]
        end
    end

    mesh = Delaunay.delaunay(xs)

    nsimplices = size(mesh.simplices, 1)
    simplices = Array{Simplex{N,Int}}(undef, nsimplices)
    for j in 1:nsimplices
        simplices[j] = Simplex(SVector{N}(mesh.simplices[j, :]))
    end

    topo = Topology(name, simplices)
    geom = Geometry(topo.name, topo, Fun{D,Pr,0}(topo, coords))
    return geom
end

export refine
function refine(name::String, geom::Geometry{D})::Geometry{D} where {D}
    @show typeof(geom.coords.values)
    return refine(name, geom.topo.nvertices, geom.topo.simplices[D],
                  geom.coords.values)
end
function refine(name::String, nvertices::Int, simplices::Simplices{N},
                coords::AbstractVector{<:Form{D,1,T}}) where {N,D,T}
    @assert N == D + 1

    for s in simplices
        @error "don't add vertices in volumes; add them on edges"
        x = sum(coords[n] for n in s.vertices) / length(s.vertices)
        x::Form{D,1}
        push!(coords, x)
    end

    @error "don't delaunay; define simplices"
    return delaunay(name, coords)
end

################################################################################

# Circumcentric (diagonal) hodge operator
function Forms.hodge(::Val{Pr}, ::Val{R}, geom::Geometry{D,T}) where {R,D,T}
    D::Int
    T::Type
    @assert 0 <= R <= D

    vol = geom.volumes[R]
    dualvol = geom.dualvolumes[R]
    @assert length(vol) == size(R, geom.topo)
    @assert length(dualvol) == size(R, geom.topo)

    return Op{D,Dl,R,Pr,R}(geom.topo,
                           Diagonal(T[dualvol[i] / vol[i]
                                      for i in 1:size(R, geom.topo)]))
end
function Forms.hodge(::Val{Dl}, ::Val{R}, geom::Geometry{D,T}) where {R,D,T}
    # return inv(hodge(Val(Pr), Val(R), geom))
    h = hodge(Val(Pr), Val(R), geom)
    h.values::Diagonal
    h1 = Op{D,Pr,R,Dl,R}(h.topo, Diagonal(inv.(h.values.diag)))
    return h1
end

# Derivatives

export coderiv
function coderiv(::Val{Pr}, ::Val{R}, geom::Geometry{D,T}) where {R,D,T}
    D::Int
    T::Type
    @assert 0 < R <= D
    op = hodge(Val(Dl), Val(R - 1), geom) *
         deriv(Val(Dl), Val(R), geom.topo) *
         hodge(Val(Pr), Val(R), geom)
    return op::Op{D,Pr,R - 1,Pr,R,T}
end

export laplace
function laplace(::Val{Pr}, ::Val{R}, geom::Geometry{D,T}) where {R,D,T}
    D::Int
    T::Type
    @assert 0 <= R <= D
    op = zero(Op{D,Pr,R,Pr,R,T}, geom.topo)
    if R > 0
        op += deriv(Val(Pr), Val(R - 1), geom.topo) *
              coderiv(Val(Pr), Val(R), geom)
    end
    if R < D
        op += coderiv(Val(Pr), Val(R + 1), geom) *
              deriv(Val(Pr), Val(R), geom.topo)
    end
    return op::Op{D,Pr,R,Pr,R,T}
end

# @computed struct Tree{D, T}
#     dom::Domain{D}
#     topo::Topology{D}
# 
#     pivot::NTuple{D, T}
#     tree::Union{NTuple{2^D, fulltype(Tree{D, T})},
#                 Vector{Int}}
# end

export coordinates
function coordinates(::Val{Pr}, ::Val{R}, geom::Geometry{D,T}) where {R,D,T}
    D::Int
    R::Int
    @assert 0 <= R <= D
    return Fun{D,Pr,R}(geom.topo,
                       map(si -> sum(geom.coords.values[si.vertices]) /
                                 length(si.vertices), geom.topo.simplices[R]))
end

################################################################################

export sample
function sample(::Val{Pr}, ::Val{R}, f::F, geom::Geometry{D,T}) where {R,F,D,T}
    D::Int
    R::Int
    T::Type
    @assert 0 <= R <= D
    U = typeof(f(zero(Form{D,1,T})))
    @assert U <: Form{D,R}
    values = map(f, coordinates(Val(Pr), Val(R), geom).values)
    Fun{D,Pr,R}(geom.topo, values)::Fun{D,Pr,R,U}
end

export project
function project(::Val{Pr}, ::Val{R}, f::F, geom::Geometry{D,T}) where {R,F,D,T}
    D::Int
    R::Int
    T::Type
    @assert 0 <= R <= D

    N = D + 1
    P = 4                       # Choice

    B = basis_products(Val(Pr), Val(R), geom)

    @assert R == 0              # TODO

    U = typeof(f(zero(Form{D,1,T})))
    @assert U <: Form{D,R}

    # Loop over all vertices
    values = Array{U}(undef, size(0, geom.topo))
    for (i, si) in enumerate(geom.topo.simplices[0])
        value = zero(U)
        # Loop over all neighbouring simplices
        # TODO: Improve this
        for (j, sj) in enumerate(geom.topo.simplices[D])
            if si.vertices[1] ∈ sj.vertices
                ss = geom.coords.values[sj.vertices]
                s = SMatrix{N,D}(ss[n][a] for n in 1:N, a in 1:D)
                X, W = simplexquad(P, collect(s))

                setup = cartesian2barycentric_setup(s)
                n = findfirst(==(si.vertices[1]), sj.vertices)
                @assert n !== nothing

                xs = SVector{N,SVector{D,T}}(convert(SVector{D,T}, ss[n])
                                             for n in 1:N)
                function kernel(x)
                    x::SVector{D,T}
                    return (basis_x(setup, n, x) * f(Form{D,1}(x)))::U
                end
                # TODO signbit???
                bf = integrate_x(kernel, Val(D), X, W)::U

                value += bf
            end
        end
        values[i] = value
    end

    values′ = map(y -> U((y,)), B \ values)

    return Fun{D,Pr,R}(geom.topo, values′)::Fun{D,Pr,R,U}
end

# TODO: evaluate many points simultaneously
export evaluate
function evaluate(geom::Geometry{D,T}, f::Fun{D,Pr,R,U},
                  x::Form{D,1,T}) where {D,R,T,U}
    D::Int
    @assert D >= 0
    R::Int
    @assert 0 <= R <= D
    @assert geom.topo == f.topo

    coords = coordinates(Val(Pr), Val(R), geom)

    # nvals = 0
    # val = zero(U)
    # # Loop over all simplices
    # # TODO: Use <https://github.com/KristofferC/NearestNeighbors.jl>
    # # TODO: Generalize this
    # @assert R == 0
    # for (i, si) in enumerate(geom.topo.simplices[D])
    #     xs = coords.values[si.vertices]
    #     xsm = SMatrix{D,D + 1}(xs[n][a] for a = 1:D, n = 1:D+1)
    #     # setup = cartesian2barycentric_setup(xsm)
    #     λ = cartesian2barycentric(xsm, convert(SVector, x))
    #     delta = sqrt(eps(T))
    #     if all(λi -> -delta <= λi <= 1 + delta, λ)
    #         # point is inside simplex
    #         fs = f.values[si.vertices]
    #         for n = 1:D+1
    #             val += fs[n] * basis_λ(n, λ)
    #         end
    #         nvals += 1
    #     end
    # end
    # # @assert nvals == 1    # there should be exactly one containing simplex
    # # return val
    # @assert nvals > 0
    # return val / nvals

    # Find nearest vertex
    i, dist = nn(geom.simplex_tree, convert(SVector, x))
    # Search all neighbouring simplices to find containing simplex
    lookup = geom.topo.lookup[D]
    rows = rowvals(lookup)
    for j0 in nzrange(lookup, i)
        j = rows[j0]
        sj = geom.topo.simplices[D][j]

        # Coordinates of simplex vertices
        xs = map(x -> convert(SVector, x), coords.values[sj.vertices])
        xs::SVector{D + 1,SVector{D,T}}
        # setup = cartesian2barycentric_setup(xs)

        # Calculate barycentric coordinates
        λ = cartesian2barycentric(xs, convert(SVector, x))
        # delta = T(0)
        delta = 10 * eps(T)
        # delta = sqrt(eps(T))
        if all(λi -> -delta <= λi <= 1 + delta, λ)
            # Function values
            fs = f.values[sj.vertices]
            # Linear interpolation
            # TODO: Use Bernstein polynomials instead
            val = sum(fs[n] * basis_λ(n, λ) for n in 1:(D + 1))
            return val
        end
    end
    @show D Pr R x
    @show geom f
    @assert false
end

function basis_products(::Val{Pr}, ::Val{R}, geom::Geometry{D,T}) where {D,T,R}
    # Check arguments
    D::Int
    R::Int
    @assert 0 <= R <= D

    N = D + 1

    # Lookup table from R-simplices to simplices
    lookup = containing_simplices(geom.topo, Val(R))
    # Lookup table from D-simplices to R-simplices
    lookup′ = sparse(transpose(lookup))

    # Result: sparse matrix
    I = Int[]
    J = Int[]
    V = T[]

    # Loop over all R-simplices
    for (i, si) in enumerate(geom.topo.simplices[R])
        # Loop over all neighbouring D-simplices
        for k in sparse_column_rows(lookup, i)
            sk = geom.topo.simplices[D][k]

            # Loop over all neighbouring R-simplices
            for j in sparse_column_rows(lookup′, k)
                sj = geom.topo.simplices[R][j]
                @assert k ∈ sparse_column_rows(lookup, j)

                # Find basis functions for simplices i and j
                @assert R == 0
                ni = findfirst(==(i), sk.vertices)
                nj = findfirst(==(j), sk.vertices)

                # Calculate overlap integral
                xs = map(x -> x.elts, geom.coords.values[sk.vertices])
                xs::SVector{N,SVector{D,T}}
                # P=2 suffices because we only have linear basis functions
                P = 2
                XS = SMatrix{N,D,T}(xs[n][d] for n in 1:N, d in 1:D)
                X, W = simplexquad(P, XS)
                setup = cartesian2barycentric_setup(xs)
                kernel(x) = basis_x(setup, ni, x) * basis_x(setup, nj, x)
                b = integrate_x(kernel, Val(D), X, W)

                push!(I, i)
                push!(J, j)
                push!(V, b)
            end
        end
    end

    n = length(geom.topo.simplices[R])
    return sparse(I, J, V, n, n)
end

function basis_products(xs::SVector{N,SVector{D,T}})::SMatrix{N,N,
                                                              T} where {N,D,T}
    D::Int
    N::Int
    @assert N == D + 1
    setup = cartesian2barycentric_setup(xs)
    XS = SMatrix{N,D,T}(xs[n][d] for n in 1:N, d in 1:D)
    P = 2   # P=2 suffices because we only have linear basis functions
    X, W = simplexquad(P, XS)
    B = zero(SMatrix{N,N,T})
    for n in 1:N, m in n:N
        f(x) = basis_x(setup, m, x) * basis_x(setup, n, x)
        bmn = integrate_x(f, Val(D), X, W)
        B = setindex(B, bmn, m, n)
    end
    for n in 1:N, m in 1:(n - 1)
        B = setindex(B, B[n, m], m, n)
    end
    return B
end

@fastmath function basis_x(setup, n::Int, x::SVector{D,T}) where {D,T}
    D::Int
    λ = cartesian2barycentric(setup, x)
    return basis_λ(n, λ)
end

# TODO: Remove this
# TODO: Use Bernstein polynomials instead
@fastmath function basis_λ(n::Int, λ::SVector{N,T}) where {N,T}
    N::Int
    @assert 1 <= n <= N
    return λ[n]
end

@fastmath function integrate_x(f::F, ::Val{D}, X::AbstractMatrix,
                               W::AbstractVector) where {F,D,T}
    D::Int
    @assert D >= 0
    @assert size(X, 2) == D
    @assert size(X, 1) == size(W, 1)

    return @inbounds begin
        s = zero(W[1] * f(SVector{D}(X[1, a] for a in 1:D)))
        for n in 1:length(W)
            s += W[n] * f(SVector{D}(X[n, a] for a in 1:D))
        end
        s
    end
end

end
