using DDF

using ComputedFieldTypes
using Grassmann
# using LinearAlgebra
# using SparseArrays
using StaticArrays
using Test



@testset "Geometry D=$D" for D in 1:Dmax
    S = Signature(D)
    V = SubManifold(S)
    T = Float64
    mf = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    dom = Domain(Chain{V,1}(sarray(T, d -> T(0), Val(D))),
                 Chain{V,1}(sarray(T, d -> T(1), Val(D))))
    xs = Fun{D, 0, fulltype(Chain{V,1,T})}(
        mf,
        fulltype(Chain{V,1,T})[
            Chain{V,1}(sarray(T, d -> T(d+1 == n), Val(D))) for n in 1:D+1])
    geom = Geometry(mf, dom, xs)
    if D==2 || D==3
        @show geom.mf.nvertices
        @show geom.mf.simplices
        for d in 1:D
            @show d geom.mf.boundaries[d]
        end
        @show geom.dom
        @show geom.coords.values
        for d in 1:D
            @show d geom.volumes[d].values
        end
        @show geom.dualcoords.values
    end
    for R in 1:D
        @test length(geom.volumes[R].values) == binomial(D+1, R+1)
    end
    @test length(geom.volumes[D].values) == 1
    @test geom.volumes[D].values[1] ≈ one(T)/factorial(D)
end

# @testset "Geometry D=$D" for D in 1:Dmax
#     S = Signature(D)
#     V = SubManifold(S)
#     T = Float64
#     @show D S V T
#     mf = DManifold(DSimplex(SVector{D+1}(1:D+1)))
#     dom = Domain{D, T}(Chain{V, 1}(sarray(T, d -> T(0), Val(D))),
#                        Chain{V, 1}(sarray(T, d -> T(1), Val(D))))
#     xs = ntuple(d -> Fun{D, 0, T}(mf, T[d == i for i in 0:D]), D)
#     cs = Coords{V, T}(mf, dom, xs)
#     hodges = hodge(cs)
# end
