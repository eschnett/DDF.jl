module Geometries

using ComputedFieldTypes
using Grassmann
using LinearAlgebra
using SparseArrays
using StaticArrays

using ..Defs
using ..Funs
using ..Manifolds
using ..Ops



# We use the following conventions:
#     D::Int               number of dimension
#     S::Signature         signature of manifold
#     V::SubManifold       manifold
#     B::DirectSum.Basis   basis
#     R::Int               rank (of multivector)
# These can have a prefix:
#     no prefix   euclidean basis (for manifold)
#     P           projective basis
#     C           conformal basis
#     (A          abstract (e.g. numbering tetrad vectors))



DVector() = Chain{D, 1} where D
DVector(D) = Chain{SubManifold(Signature(D)), 1}
DVector(D, T) = fulltype(Chain{SubManifold(Signature(D)), 1, T})

export Domain
@computed struct Domain{D, T}
    xmin::DVector(D, T)
    xmax::DVector(D, T)
    function Domain{D,T}(xmin::Chain{V,1,T}, xmax::Chain{V,1,T}) where {D,V,T}
        V::SubManifold
        T::Type
        @assert D == ndims(V)
        new{D,T}(xmin, xmax)
    end
    Domain(xmin::Chain{V,1,T}, xmax::Chain{V,1,T}) where {V,T} =
        Domain{ndims(V),T}(xmin, xmax)
end

Defs.invariant(dom::Domain) = true



export Geometry
@computed struct Geometry{D, T}
    mf::DManifold{D}
    dom::fulltype(Domain{D, T})
    # Coordinates of vertices
    coords::Fun{D, Pr, 0, DVector(D, T)}
    # Volumes of 0-forms are always 1 and don't need to be stored
    volumes::Dict{Int, Fun{D, Pr, R, T} where R}
    # Coordinates of vertices of dual grid, i.e. circumcentres of top
    # simplices
    dualcoords::Fun{D, Dl, D, DVector(D, T)}
    # Dual volumes of dual top-forms are always 1 and don't need to be stored
    # dualvolumes[R = D-DR]
    dualvolumes::Dict{Int, Fun{D, Dl, R, T} where R}

    function Geometry{D, T}(mf::DManifold{D},
                            dom::Domain{D, T},
                            coords::Fun{D, Pr, 0, Chain{V, 1, T, X1}},
                            volumes::Dict{Int, Fun{D, Pr, R, T} where R},
                            dualcoords::Fun{D, Dl, D, Chain{V, 1, T, X1}},
                            dualvolumes::Dict{Int, Fun{D, Dl, R, T} where R}
                            ) where {D, T, V, X1}
        D::Int
        T::Type
        @assert D >= 0
        @assert isempty(symdiff(keys(volumes), 0:D))
        @assert isempty(symdiff(keys(dualvolumes), 0:D))
        geom = new(mf, dom, coords, volumes, dualcoords, dualvolumes)
        @assert invariant(geom)
        geom
    end
end



function Defs.invariant(geom::Geometry{D})::Bool where {D}
    invariant(geom.mf) || return false
    invariant(geom.dom) || return false
    isempty(symdiff(keys(geom.volumes), 0:D)) || return false
    isempty(symdiff(keys(geom.dualvolumes), 0:D)) || return false
    return true
end



function Geometry(mf::DManifold{D},
                  dom::Domain{D, T},
                  coords::Fun{D, Pr, 0, Chain{V, 1, T, X1}}
                  ) where {D, T, V, X1}
    D::Int
    T::Type
    V::SubManifold
    @assert ndims(V) == D

    # Calculate volumes
    volumes = Dict{Int, Fun{D,Pr,R,T} where {R}}()
    for R in 0:D
        values = Array{T}(undef, size(R, mf))
        for (i,s) in enumerate(mf.simplices[R])
            cs = coords.values[s.vertices]
            xs = SVector{R,fulltype(Chain{V,1,T})}(cs[n+1] - cs[1] for n in 1:R)
            if length(xs) == 0
                vol = one(T)
            elseif length(xs) == 1
                vol = scalar(abs(xs[1])).v
            else
                vol = scalar(abs(∧(xs...))).v
            end
            vol /= factorial(R)
            vol *= bitsign(s.signbit)
            values[i] = vol
        end
        @assert all(>(0), values)
        vols = Fun{D, Pr, R, T}(mf, values)
        volumes[R] = vols
    end

    # Calculate circumcentres
    values = Array{fulltype(Chain{V,1,T})}(undef, size(D, mf))
    for R in D:D
        for (i,s) in enumerate(mf.simplices[R])
            cs = coords.values[s.vertices]
            cc = circumcentre(cs)
            values[i] = cc
        end
    end
    dualcoords = Fun{D, Dl, D, fulltype(Chain{V,1,T})}(mf, values)

    # Check Delaunay condition:
    # No vertex must lie in (or on) the circumcentre of a simplex
    for (i,si) in enumerate(mf.simplices[D])
        xi1 = coords[si.vertices[1]]
        cc = dualcoords.values[i]
        cr2 = scalar(abs2(xi1 - cc)).v
        # TODO: Check only vertices of neighbouring simplices
        for (j,sj) in enumerate(mf.simplices[0])
            if j ∉ si.vertices
                xj = scalar(abs(coords[sj.vertices[1]]))
                d2 = scalar(abs2(xj - cc)).v
                @assert d2 > cr2 + sqrt(eps(T))
            end
        end
    end

    # Check one-sidedness for boundary simplices:
    # TODO

    # Check that all circumcentres lie inside their simplices
    for (i,si) in enumerate(mf.simplices[D])
        xsi = coords[si.vertices]
        N = length(xsi)
        cc = dualcoords.values[i]
        svol = sign(scalar(∧(xsi...)).v)
        for a in 1:N
            xsj = SVector{N}(b==a ? cc : xsi[b] for b in 1:N)
            @assert sign(scalar(∧(xsj...)).v) == svol
        end
    end

    # Calculate circumcentric dual volumes
    # [1198555.1198667, page 5]
    dualvolumes = Dict{Int, Fun{D,Dl,R,T} where {R}}()
    for R in D:-1:0
        if R == D
            values = ones(T, size(R, mf))
        else
            bnds = mf.boundaries[R+1]
            values = zeros(T, size(R, mf))
            sis = mf.simplices[R]::Vector{DSimplex{R+1, Int}}
            sjs = mf.simplices[R+1]::Vector{DSimplex{R+2, Int}}
            for (i,si) in enumerate(sis)
                # TODO: This is expensive
                js = findnz(bnds[i,:])[1]
                for j in js
                    sj = sjs[j]
                    b = dualvolumes[R+1][j]
                    # TODO: Calculate lower-rank circumcentres as
                    # intersection between boundary and the line
                    # connecting two simplices?
                    # TODO: Cache circumcentres ahead of time
                    @assert length(si.vertices) == R+1
                    @assert length(sj.vertices) == R+2
                    xsi = coords[si.vertices]
                    cci = circumcentre(xsi)
                    xsj = coords[sj.vertices]
                    ccj = circumcentre(xsj)
                    # TODO: Handle case where the volume should be
                    # negative (i.e. when the volume circumcentre ccj
                    # is on the "other" side of the face circumcentre
                    # cci)
                    h = scalar(abs(cci - ccj)).v
                    values[i] += b * h / factorial(D-R)
                end
            end
        end
        @assert all(>(0), values)
        vols = Fun{D, Dl, R, T}(mf, values)
        dualvolumes[R] = vols
    end

    # # Calculate barycentres
    # values = Array{fulltype(Chain{V,1,T})}(undef, size(D, mf))
    # for R in D:D
    #     for (i,s) in enumerate(mf.simplices[R])
    #         cs = sarray(fulltype(Chain{V,1,T}),
    #                     n -> coords.values[s.vertices[n]],
    #                     Val(R+1))
    #         cc = +(cs...) / length(cs)
    #         values[i] = cc
    #     end
    # end
    # dualcoords = Fun{D, Dl, D, fulltype(Chain{V,1,T})}(mf, values)
    #
    # # Calculate barycentric dual volumes
    # dualvolumes = Dict{Int, Fun{D,Dl,R,T} where {R}}()
    # for DR in 0:D
    #     R = D - DR
    #     if R == D
    #         values = ones(T, size(R, mf))
    #     else
    #         sjs = mf.simplices[R+1]::Vector{DSimplex{R+2, Int}}
    #         bnds = mf.boundaries[R+1]
    #         values = zeros(T, size(R, mf))
    #         # Loop over all duals of rank R (e.g. faces)
    #         if R == 0
    #             sis = (DSimplex{1,Int}(SVector(i)) for i in 1:mf.nvertices)
    #         else
    #             sis = mf.simplices[R]::Vector{DSimplex{R+1, Int}}
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
    #                 vol = scalar(abs(∧(ysi...))).v
    #                 # TODO: take sign into account?
    #                 @assert vol > 0
    #                 values[i] += vol
    #             end
    #         end
    #     end
    #     @assert all(>(0), values)
    #     vols = Fun{D, Dl, R, T}(mf, values)
    #     dualvolumes[R] = vols
    # end

    Geometry{D, T}(mf, dom, coords, volumes, dualcoords, dualvolumes)
end



export hodge

# Circumcentric (diagonal) hodge operator
function hodge(::Val{Pr}, ::Val{R}, geom::Geometry{D, T}) where {R, D, T}
    D::Int
    T::Type
    @assert 0 <= R <= D
    S = Signature(D)
    V = SubManifold(S)

    vol = geom.volumes[R]
    dualvol = geom.dualvolumes[R]
    @assert length(vol) == size(R, geom.mf)
    @assert length(dualvol) == size(R, geom.mf)
    
    Op{D, Dl, R, Pr, R}(
        geom.mf,
        Diagonal(T[dualvol[i] / vol[i] for i in 1:size(R, geom.mf)]))
end
hodge(::Val{Dl}, ::Val{R}, geom::Geometry{D, T}) where {R, D, T} =
    inv(hodge(Val(Pr), Val(R), geom))



# Derivatives

export coderiv
function coderiv(::Val{Pr}, ::Val{R}, geom::Geometry{D, T}) where {R, D, T}
    D::Int
    T::Type
    @assert 0 < R <= D
    op = hodge(Val(Dl), Val(R-1), geom) *
        deriv(Val(Dl), Val(R), geom.mf) *
        hodge(Val(Pr), Val(R), geom)
    op::Op{D, Pr, R-1, Pr, R, T}
end

export laplace
function laplace(::Val{Pr}, ::Val{R}, geom::Geometry{D, T}) where {R, D, T}
    D::Int
    T::Type
    @assert 0 <= R <= D
    op = zero(Op{D, Pr, R, Pr, R, T}, geom.mf)
    if R > 0
        op += deriv(Val(Pr), Val(R-1), geom.mf) * coderiv(Val(Pr), Val(R), geom)
    end
    if R < D
        op += coderiv(Val(Pr), Val(R+1), geom) * deriv(Val(Pr), Val(R), geom.mf)
    end
    op::Op{D, Pr, R, Pr, R, T}
end



# @computed struct Tree{D, T}
#     dom::Domain{D}
#     mf::DManifold{D}
# 
#     pivot::NTuple{D, T}
#     tree::Union{NTuple{2^D, fulltype(Tree{D, T})},
#                 Vector{Int}}
# end



# function sample
# function evaluate

end
