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



function volumes_type(D::Int, ::Type{T})::Type where {T}
    Tuple{(fulltype(Fun{D, R, T}) for R in 1:D)...}
end

export Geometry
@computed struct Geometry{V, T}
    mf::DManifold{ndims(V)}
    dom::Domain{V, T}
    coords::Fun{ndims(V), 0, fulltype(Chain{V, 1, T})}
    # Volumes of 0-forms are always 1 and are not stored
    volumes::volumes_type(ndims(V), T)
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
    volumes = Fun{D}[]
    @show D V T S B
    for R in 1:D
        @show R
        values = Array{T}(undef, size(R, mf))
        for (i,s) in enumerate(mf.simplices[R])
            @show i s
            cs = sarray(fulltype(Chain{V, 1, T}),
                        n -> coords.values[s.vertices[n]],
                        Val(R+1))
            @show cs
            xs = sarray(fulltype(Chain{V, 1, T}), n -> cs[n+1] - cs[1], Val(R))
            @show xs
            if length(xs) == 1
                vol = abs(xs[1])
            else
                vol = abs(∧(xs...))
            end
            @show vol
            vol /= factorial(R)
            @show vol
            @assert isscalar(vol)
            vol = scalar(vol).v
            @show typeof(vol)
            @show vol
            dump(vol)
            vol::T
            vol *= bitsign(s.signbit)
            vol::T
            values[i] = vol
        end
        vols = Fun{D, R, T}(mf, values)
        push!(volumes, vols)
        @show vols
    end
    @show volumes
    volumes1 = tuple(volumes...)
    @show volumes1

    Geometry{V, T}(mf, dom, coords, volumes1)
end



# export circumcentre
# function circumcentre(xs0::SVector{R, <:SVector{D, T}}) where {R, D, T}
#     S = Signature(D)
#     V = SubManifold(S)
#     B = Λ(S)
#     xs = sarray(fulltype(Chain{V, 1, T}),
#                 i -> Chain{V, 1}(xs0[i]),
#                 Val(R))
#     cc = circumcentre(xs)
#     cc0 = cc.v
#     cc0
# end
# 
# # function q()
# # julia> using Grassmann
# # julia> S=Signature(2)
# # julia> V=SubManifold(S)
# # julia> B=Λ(S)
# # julia> x0=Chain{V,1}(0,0)
# # julia> x1=Chain{V,1}(1,0)
# # julia> x2=Chain{V,1}(0,1)
# # 
# # julia> CS=Signature(4,1,1)
# # julia> CV=SubManifold(CS)
# # julia> CB=Λ(CS)
# # julia> cx0=Chain{CV,1}(0,0,0,0)
# # julia> cx1=Chain{CV,1}(0,0,1,0)
# # julia> cx2=Chain{CV,1}(0,0,0,1)
# # julia> cx0=↑(cx0)
# # julia> cx1=↑(cx1)
# # julia> cx2=↑(cx2)
# # julia> @assert isvector(cx0)
# # julia> @assert isvector(cx1)
# # julia> @assert isvector(cx2)
# # julia> cx0=vector(cx0)
# # julia> cx1=vector(cx1)
# # julia> cx2=vector(cx2)
# # 
# # julia> Vol=⋆∧(cx0,cx1,cx2)
# # julia> cc=↓(Vol)
# # julia> @assert isvector(cc)
# # julia> cc=vector(cc)
# # julia> r2=abs2(cc)-2Vol[1]
# # julia> @assert isscalar(r2)
# # julia> r2=scalar(r2)
# # julia> abs2(cx0-cc)==r2
# # julia> abs2(cx1-cc)==r2
# # julia> abs2(cx2-cc)==r2
# # end
# 
# # @error "need to go to lower dimension"
# function circumcentre(xs::SVector{R, <:Chain{V, 1, T}}) where {R, V, T}
#     # G. Westendorp, A formula for the N-circumsphere of an N-simplex,
#     # <https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm>,
#     # April 2013.
#     @assert !hasinf(V)
#     @assert !hasorigin(V)
#     S = Signature(V)
#     D = ndims(S)
#     @assert R == D + 1
# 
#     CS = Signature(D+2, 1, 1)
#     CD = ndims(CS)
#     CV = SubManifold(CS)
#     CB = Λ(CS)
# 
#     # Convert Euclidean to conformal basis
#     cxs = sarray(fulltype(Chain{CV, 1, T}), i -> conformal(xs[i]), Val(R))
#     # Circumsphere (this formula is why we are using conformal GA)
#     X = ∧(cxs)
#     # Hodge dual
#     sX = ⋆X
#     sX /= sX.v[2]
#     @assert sX.v[2] == 1
#     # Euclidean part is centre
#     cc = Chain{V,1}(sarray(T, i -> sX.v[i+2], Val(D)))
#     # Calculate radius
#     r2 = scalar(abs2(cc)).v - 2 * sX.v[1]
# 
#     # Check radii
#     for i in 1:R
#         ri2 = scalar(abs2(xs[i] - cc)).v
#         @assert abs(ri2 - r2) <= T(1.0e-12) * r2
#     end
# 
#     cc::fulltype(Chain{V, 1, T})
# end
# 
# 
# 
# export circumcentres
# function circumcentres(geom::Geometry{V, T}) where {V, T}
#     D = ndims(V)
#     mf = geometry.mf
#     n = dim(Val(D), mf)
#     ccs = Array{SVector{D, T}}(undef, n)
#     if D == 0
#         return ccs
#     end
#     simplices = mf.simplices[D]
#     for (i,s) in enumerate(simplices)
#         # cs[a][b] = geometry of vertex a of this simplex
#         cs = sarray(SVector{D, T},
#                     j -> sarray(T, b -> geom.geometry[b][s[j]], Val(D)),
#                     Val(D+1))
#         ccs[i] = circumcentre(cs)
#         # TODO: Check Delauney condition
#     end
#     ccs::Vector{SVector{D, T}}
# end
# 
# 
# 
# export hodge
# function hodge(geom::Geometry{V, T}) where {V, T}
#     @show "hodge" geometry
#     S = Signature(V)
#     D = ndims(V)
#     mf = geometry.mf
# 
#     # ccs = circumcentres(geometry)
# 
#     dualVols = Array{Vector{T}}(undef, max(0, D-1))
# 
#     for R in D-1:-1:1
#         @show R
#         R1 = R + 1
#     
#         dualVol1 = R1 == D ? nothing : dualVols[R1]
#         dualVol = Array{T}(undef, dim(Val(R), mf))
#     
#         # TODO: use boundary operator to find connectivity
#         for (i,si) in enumerate(mf.simplices[R])
#             @show i si
#             @assert length(si) == R+1
#             xis = sarray(SVector{D,T},
#                          k -> sarray(T, a -> geometry.geometry[a][si[k]], Val(D)),
#                          Val(R+1))
#             @show xis
#             cci = circumcentre(xis)
#             Vol = T(0)
#             for (j,sj) in enumerate(mf.simplices[R1])
#                 @show j sj
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
