module Geometries

using ComputedFieldTypes
using Grassmann
using SparseArrays
using StaticArrays

using ..Defs
using ..Funs
using ..Manifolds



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
    coords::Fun{D, 0, DVector(D, T)}
    # Volumes of 0-forms are always 1 and are not stored
    volumes::Dict{Int, Fun{D, R, T} where R}
    # Coordinates of vertices of dual grid, i.e. circumcentres of top
    # simplices
    dualcoords::Fun{D, D, DVector(D, T)}
    # Dual volumes of dual top-forms are always 1 and are not stored
    # dualvolumes[R = D-DR]
    dualvolumes::Dict{Int, Fun{D, R, T} where R}
end



function Defs.invariant(geom::Geometry{D})::Bool where {D}
    invariant(geom.mf) || return false
    invariant(geom.dom) || return false
    isempty(symdiff(keys(geom.volumes), 1:D)) || return false
    isempty(symdiff(keys(geom.dualvolumes), 0:D-1)) || return false
    return true
end



function Geometry(mf::DManifold{D},
                  dom::Domain{D, T},
                  coords::Fun{D, 0, Chain{V, 1, T, X1}}
                  ) where {D, T, V, X1}
    D::Int
    T::Type
    V::SubManifold
    @assert ndims(V) == D

    # TODO: Check Delauney criterion

    # Calculate volumes
    volumes = Dict{Int, Fun{D,R,T} where {R}}()
    for R in 1:D
        values = Array{T}(undef, size(R, mf))
        for (i,s) in enumerate(mf.simplices[R])
            cs = sarray(fulltype(Chain{V, 1, T}),
                        n -> coords.values[s.vertices[n]],
                        Val(R+1))
            xs = sarray(fulltype(Chain{V, 1, T}), n -> cs[n+1] - cs[1], Val(R))
            if length(xs) == 1
                vol = abs(xs[1])
            else
                vol = abs(∧(xs...))
            end
            vol /= factorial(R)
            @assert isscalar(vol)
            vol = scalar(vol).v::T
            vol *= bitsign(s.signbit)
            values[i] = vol
        end
        vols = Fun{D, R, T}(mf, values)
        volumes[R] = vols
    end

    # Calculate circumcentres
    values = Array{fulltype(Chain{V,1,T})}(undef, size(D, mf))
    for R in D:D
        for (i,s) in enumerate(mf.simplices[R])
            cs = sarray(fulltype(Chain{V, 1, T}),
                        n -> coords.values[s.vertices[n]],
                        Val(R+1))
            cc = circumcentre(cs)
            values[i] = cc
        end
    end
    circumcentres = Fun{D, D, fulltype(Chain{V,1,T})}(mf, values)

    # Calculate dual volumes
    # [1198555.1198667, page 5]
    dualvolumes = Dict{Int, Fun{D,R,T} where {R}}()
    for DR in 1:D
        R = D - DR
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
                b = R+1 == D ? one(T) : dualvolumes[R+1+1][j]
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
        vols = Fun{D, R, T}(mf, values)
        dualvolumes[R+1] = vols
    end

    Geometry{D, T}(mf, dom, coords, volumes, circumcentres, dualvolumes)
end



export hodge
function hodge(geom::Geometry{D, T}) where {D, T}
    D::Int
    T::Type
    S = Signature(D)
    V = SubManifold(S)

    nothing

#     # @show "hodge" geometry
#     S = Signature(V)
#     D = ndims(V)
#     mf = geometry.mf
# 
#     # ccs = circumcentres(geometry)
# 
#     dualVols = Array{Vector{T}}(undef, max(0, D-1))
# 
#     for R in D-1:-1:1
#         # @show R
#         R1 = R + 1
#     
#         dualVol1 = R1 == D ? nothing : dualVols[R1]
#         dualVol = Array{T}(undef, dim(Val(R), mf))
#     
#         # TODO: use boundary operator to find connectivity
#         for (i,si) in enumerate(mf.simplices[R])
#             # @show i si
#             @assert length(si) == R+1
#             xis = sarray(SVector{D,T},
#                          k -> sarray(T, a -> geometry.geometry[a][si[k]], Val(D)),
#                          Val(R+1))
#             # @show xis
#             cci = circumcentre(xis)
#             Vol = T(0)
#             for (j,sj) in enumerate(mf.simplices[R1])
#                 # @show j sj
#                 if any(sj[k] ==i for k in 1:length(sj))
#                     b = R1 == D ? T(1) : dualVol1[j]
#                     @assert length(sj) == R1+1
#                     xjs = sarray(
#                         SVector{D,T},
#                         k -> sarray(T, a -> geometry.geometry[a][sj[k]], Val(D)),
#                         Val(R1+1))
#                     ccj = circumcentre(xjs)
#                     h = sqrt(sum(T, a -> (cci[a] - ccj[a])^2, Val(D)))
#                     Vol += b * h / (D - R)
#                 end
#             end
#             dualVol[i] = Vol
#         end
#     
#         dualVols[R] = dualVol
#     end
# 
#     tuple(dualVols...)::NTuple{max(0, D-1), Vector{T}}
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
