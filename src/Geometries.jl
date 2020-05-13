module Geometries

using ComputedFieldTypes
using Grassmann
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



export Domain
@computed struct Domain{V, T}
    xmin::fulltype(Chain{V, 1, T})
    xmax::fulltype(Chain{V, 1, T})
    Domain{V,T}(xmin::Chain{V,1,T}, xmax::Chain{V,1,T}) where {V,T} =
        new{V,T}(xmin, xmax)
    Domain(xmin::Chain{V,1,T}, xmax::Chain{V,1,T}) where {V,T} =
        Domain{V,T}(xmin, xmax)
end



volumes_type(D::Int, ::Type{T}) where {T} =
    Tuple{(fulltype(Fun{D, R, T}) for R in 1:D)...}
circumvolumes_type(D::Int, ::Type{T}) where {T} =
    Tuple{(fulltype(Fun{D, R, T}) for R in 0:D-1)...}

export Geometry
@computed struct Geometry{V, T}
    mf::DManifold{ndims(V)}
    dom::Domain{V, T}
    coords::Fun{ndims(V), 0, fulltype(Chain{V, 1, T})}
    # Volumes of 0-forms are always 1 and are not stored
    volumes::volumes_type(ndims(V), T)
    # Coordinates of dual grid, i.e. circumcentres of top simplices
    dualcoords::Fun{ndims(V), ndims(V), fulltype(Chain{V, 1, T})}
end



function Geometry(mf::DManifold{D},
                  dom::Domain{V, T},
                  coords::Fun{D, 0, <:Chain{V, 1, T}}) where {D, V, T}
    D::Int
    V::SubManifold
    @assert D == ndims(V)
    T::Type

    S = Signature(V)
    B = Λ(S)

    # Calculate volumes
    volumes = (Fun{D,R,T} where {R})[]
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
        push!(volumes, vols)
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

    Geometry{V, T}(mf, dom, coords, tuple(volumes...), circumcentres)
end



function circumcentre(xs::SVector{R, <:Chain{V, 1, T}}) where {R, V, T}
    # G. Westendorp, A formula for the N-circumsphere of an N-simplex,
    # <https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm>,
    # April 2013.
    @assert iseuclidean(V)
    D = ndims(V)
    @assert R == D + 1

    # Convert Euclidean to conformal basis
    cxs = map(conformal, xs)
    # Circumsphere (this formula is why we are using conformal GA)
    X = ∧(cxs)
    # Hodge dual
    sX = ⋆X
    # Euclidean part is centre
    cc = euclidean(sX)

    # Calculate radius
    r2 = scalar(abs2(cc)).v - 2 * sX.v[1]
    # Check radii
    for i in 1:R
        ri2 = scalar(abs2(xs[i] - cc)).v
        @assert abs(ri2 - r2) <= T(1.0e-12) * r2
    end

    cc::Chain{V, 1, T}
end



# export hodge
# function hodge(geom::Geometry{V, T}) where {V, T}
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
# end



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
