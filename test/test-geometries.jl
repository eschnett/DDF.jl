using DDF

using ComputedFieldTypes
using Grassmann
# using LinearAlgebra
# using SparseArrays
using StaticArrays
using Test



#TODO @testset "Geometry D=$D" for D in 1:Dmax
@testset "Geometry D=$D" for D in 1:2
    S = Signature(D)
    V = SubManifold(S)
    T = Float64
    mf = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    dom = Domain(Chain{V,1}(sarray(T, d -> T(0), Val(D))),
                 Chain{V,1}(sarray(T, d -> T(1), Val(D))))
    xs = Fun{D, 0, fulltype(Chain{V,1,T})}(
        mf,
        fulltype(Chain{V,1,T})[
            Chain{V,1}(sarray(T, d -> d+1 == n, Val(D)))
            for n in 1:D+1])
    geom = Geometry(mf, dom, xs)
    if D==2
        @show geom.mf.nvertices
        @show geom.mf.simplices
        @show geom.mf.boundaries[1]
        @show geom.mf.boundaries[2]
        @show geom.dom
        @show geom.coords.values
        @show geom.volumes[1].values
        @show geom.volumes[2].values
    end
    # ccs = circumcentres(geom)
end

# @testset "Geometry D=$D" for D in 1:Dmax
#     S = Signature(D)
#     V = SubManifold(S)
#     T = Float64
#     @show D S V T
#     mf = DManifold(DSimplex(SVector{D+1}(1:D+1)))
#     dom = Domain{V, T}(Chain{V, 1}(sarray(T, d -> T(0), Val(D))),
#                        Chain{V, 1}(sarray(T, d -> T(1), Val(D))))
#     xs = ntuple(d -> Fun{D, 0, T}(mf, T[d == i for i in 0:D]), D)
#     cs = Coords{V, T}(mf, dom, xs)
#     hodges = hodge(cs)
# end
