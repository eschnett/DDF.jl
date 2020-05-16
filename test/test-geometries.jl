using DDF

using ComputedFieldTypes
using Grassmann
using StaticArrays
using Test



@testset "Geometry D=$D" for D in 1:Dmax
    S = Signature(D)
    V = SubManifold(S)
    T = Float64
    mf = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    dom = Domain(Chain{V,1}(zeros(SVector{D,T})),
                 Chain{V,1}(ones(SVector{D,T})))
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



@testset "Geometry ops D=$D R=$R" for D in 1:Dmax, R in 0:D
    S = Signature(D)
    V = SubManifold(S)
    T = Float64

    dom = Domain(Chain{V,1}(zeros(SVector{D,T})),
                 Chain{V,1}(ones(SVector{D,T})))

    # Empty manifold
    mf0 = DManifold(Val(D))
    xs0 = Fun{D, 0, fulltype(Chain{V,1,T})}(mf0, fulltype(Chain{V,1,T})[])
    geom0 = Geometry(mf0, dom, xs0)

    # Simplex
    mf1 = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    xs1 = Fun{D, 0, fulltype(Chain{V,1,T})}(
        mf1,
        [Chain{V,1}(sarray(T, d -> T(d+1 == n), Val(D))) for n in 1:D+1])
    geom1 = Geometry(mf1, dom, xs1)

    # Cube
    mf2 = hypercube_manifold(Val(D))
    xs2 = Fun{D, 0, fulltype(Chain{V,1,T})}(
        mf2,
        [Chain{V,1}(sarray(T, d -> T((n-1) & (1<<(d-1)) != 0), Val(D)))
         for n in 1:1<<D])
    geom2 = Geometry(mf2, dom, xs2)

    geoms = [geom0, geom1, geom2]

    for geom in geoms

        for R in 0:D
            f0 = ones(Fun{D, R, T}, geom.mf)
            f1 = id(Fun{D, R, T}, geom.mf)
            fs = [f0, f1]

            for f in fs
                h = hodge(Val(R), geom)
                hf = h*f
                hf::Fun{D, R, T}
            end

            #TODO if R > 0
            #TODO     b = coderiv(Val(R), mf)
            #TODO     for f in fs
            #TODO         cf = c*f
            #TODO         cf::Fun{D, R-1, T}
            #TODO     end
            #TODO end
            #TODO 
            #TODO for f in fs
            #TODO     l = laplace(Val(R), geom)
            #TODO     lf = l*f
            #TODO     hf::Fun{D, R, T}
            #TODO end
        end
    end
end
