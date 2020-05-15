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
    S = Signature(D)
    B = Λ(S)

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

    # # Calculate dual mesh
    # # TODO: Move this to "DManifold"
    # dualcells = Vector{Vector{T}}[] # dualcells[DR][icell][ivertex] = vertex
    # ndualvertices = size(D, mf)
    # for DR in 1:D
    #     # For each primal R-simplex, find all neighbours
    #     bnds = mf.boundaries[DR]
    #     dss = Array{Vector{T}(undef, size(DR,mf))
    #     for (i,si) in enumerate(mf.simplices[DR])
    #         # Find all neighbours (which share a boundary) of this
    #         # simplex i
    #         # TODO: Use SVector
    #         neighbours = Int[]
    #         for b in findnz(bnds[:,i])[1]
    #             # Loop over all simplices that share this boundary,
    #             # excluding i
    #             # TODO: This is probably slow?
    #             js = findnz(bnds[b,:])[1]
    #             @show i b js
    #             @assert length(js) ∈ 1:2 # 1 at boundary, 2 in interior
    #             @assert i ∈ js
    #             for j in js
    #                 if j != i
    #                     sj = mf.simplices[DR][j]
    # 
    #                     # Check vertex sets for conssitency
    #                     vimj = setdiff(si.vertices, sj.vertices)
    #                     vjmi = setdiff(sj.vertices, si.vertices)
    #                     @assert length(vimj) == 1
    #                     @assert length(vjmi) == 1
    #                     @assert vimj[1] != vjmi[1]
    # 
    #                     push!(neighbours, j)
    #                 end
    #             end
    #         end
    # 
    #         @show DR i neighbours
    #         @error "this does not hold at boundaries -- what to do?"
    #         @error "the dual of a simplicial complex is a cell complex!"
    #         @assert length(neighbours) == DR+1
    #         dvs = SVector{DR+1}(neighbours)
    #         # TODO: calculate sign by looking at circumcentre
    #         # (see arXiv:1103.3076v2 [cs.NA], section 10)
    #         ds = DSimplex(dvs, true)
    #         dss[i] = ds
    #     end
    #     push!(dualsimplices, dss)
    # end

    # Calculate dual volumes
    # [1198555.1198667, page 5]
    dualvolumes = Dict{Int, Fun{D,R,T} where {R}}()
    for R in D-1:-1:0
        values = zeros(T, size(R, mf))
        #TODO for (i,si) in enumerate(mf.simplices[R])
        #TODO     # TODO: This is expensive
        #TODO     js = findnz(mf.boundary[R+1][i,:])[1]
        #TODO     for j in js
        #TODO         sj = mf.simplices[R+1][j]
        #TODO         b = R+1 == D ? one(T) : dualvolumes[R+1+1][j]
        #TODO         # TODO: Calculate lower-rank circumcentres as
        #TODO         # intersection between boundary and the line
        #TODO         # connecting two simplices?
        #TODO         h = abs(circumcentre(si.vertices) - circumcentre(sj.vertices))
        #TODO         values[j] += b * h / factorial(D-R)
        #TODO     end
        #TODO end
        vols = Fun{D, R, T}(mf, values)
        dualvolumes[R+1] = vols
    end

    Geometry{D, T}(mf, dom, coords, volumes, circumcentres, dualvolumes)
end



# export hodge
# function hodge(geom::Geometry{D, T}) where {V, T}
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
