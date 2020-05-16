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

    # TODO: Check Delauney criterion

    # Calculate volumes
    volumes = Dict{Int, Fun{D,Pr,R,T} where {R}}()
    for R in 0:D
        values = Array{T}(undef, size(R, mf))
        for (i,s) in enumerate(mf.simplices[R])
            cs = sarray(fulltype(Chain{V,1,T}),
                        n -> coords.values[s.vertices[n]],
                        Val(R+1))
            xs = sarray(fulltype(Chain{V,1,T}), n -> cs[n+1] - cs[1], Val(R))
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
        vols = Fun{D, Pr, R, T}(mf, values)
        volumes[R] = vols
    end

    # Calculate circumcentres
    values = Array{fulltype(Chain{V,1,T})}(undef, size(D, mf))
    for R in D:D
        for (i,s) in enumerate(mf.simplices[R])
            cs = sarray(fulltype(Chain{V,1,T}),
                        n -> coords.values[s.vertices[n]],
                        Val(R+1))
            cc = circumcentre(cs)
            values[i] = cc
        end
    end
    circumcentres = Fun{D, Dl, D, fulltype(Chain{V,1,T})}(mf, values)

    # Calculate dual volumes
    # [1198555.1198667, page 5]
    dualvolumes = Dict{Int, Fun{D,Dl,R,T} where {R}}()
    for DR in 0:D
        R = D - DR
        if R == D
            values = ones(T, size(R, mf))
        else
            bnds = mf.boundaries[R+1]
            values = zeros(T, size(R, mf))
            if R == 0
                sis = (DSimplex{1,Int}(SVector(i)) for i in 1:mf.nvertices)
            else
                sis = mf.simplices[R]::Vector{DSimplex{R+1, Int}}
            end
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
                    xsi = sarray(fulltype(Chain{V, 1, T}),
                                 n -> coords[si.vertices[n]],
                                 Val(R+1))
                    cci = circumcentre(xsi)
                    xsj = sarray(fulltype(Chain{V, 1, T}),
                                 n -> coords[sj.vertices[n]],
                                 Val(R+2))
                    ccj = circumcentre(xsj)
                    h = scalar(abs(cci - ccj)).v
                    values[i] += b * h / factorial(DR)
                end
            end
        end
        vols = Fun{D, Dl, R, T}(mf, values)
        dualvolumes[R] = vols
    end

    Geometry{D, T}(mf, dom, coords, volumes, circumcentres, dualvolumes)
end



export hodge
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

# Derivative

export coderiv
function coderiv(::Val{Pr}, ::Val{R}, geom::Geometry{D, T}) where {R, D, T}
    D::Int
    T::Type
    @assert 0 < R <= D
    op = hodge(Val(Dl), Val(R-1), geom) *
        dualderiv(Val(Dl), Val(R), geom.mf) *
        hodge(Val(Pr), Val(R), geom)
    op::Op{D, Pr, R-1, Pr, R, T}
end

export laplace
function laplace(::Val{Pr}, ::Val{R}, geom::Geometry{D, T}) where {R, D, T}
    D::Int
    T::Type
    @assert 0 <= R <= D
    op = zero(Op{D, R, R, T}, geom.mf)
    if R > 0
        op += deriv(Val(Pr), Val(R-1), geom) * coderiv(Val(Pr), Val(R), geom)
    end
    if R < D
        op += coderiv(Val(Pr), Val(R+1), geom) * deriv(Val(Pr), Val(R), geom)
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
