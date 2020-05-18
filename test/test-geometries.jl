using DDF

using ComputedFieldTypes
using Grassmann
using StaticArrays
using Test



@testset "Geometry D=$D" for D in 1:Dmax
    S = Signature(D)
    V = SubManifold(S)
    T = Float64

    dom = Domain(Chain{V,1}(zeros(SVector{D,T})),
                 Chain{V,1}(ones(SVector{D,T})))

    @testset "Empty manifold" begin
        mf = DManifold(Val(D))
        xs = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(mf, fulltype(Chain{V,1,T})[])
        geom = Geometry(mf, dom, xs)

        for R in 0:D
            @test length(geom.volumes[R].values) == 0
        end
    end



    # Orthogonal simplices are not Delaunay
    # @testset "Orthogonal simplex" begin
    #     mf = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    #     xs = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #         mf,
    #         [Chain{V,1}(SVector{D}(T(d+1 == n) for d in 1:D)) for n in 1:D+1])
    #     geom = Geometry(mf, dom, xs)
    # 
    #     for R in 0:D
    #         @test length(geom.volumes[R].values) == binomial(D+1, R+1)
    #     end
    #     @test geom.volumes[D].values[1] ≈ one(T)/factorial(D)
    # 
    #     # if D==2 || D==3
    #     #     @show geom.mf.nvertices
    #     #     @show geom.mf.simplices
    #     #     for d in 1:D
    #     #         @show d geom.mf.boundaries[d]
    #     #     end
    #     #     @show geom.dom
    #     #     @show geom.coords.values
    #     #     for d in 1:D
    #     #         @show d geom.volumes[d].values
    #     #     end
    #     #     @show geom.dualcoords.values
    #     # end
    # end


    @testset "Regular simplex" begin
        mf = DManifold(DSimplex(SVector{D+1}(1:D+1)))
        #REMOVE xs = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
        #REMOVE     mf, regular_simplex(Val(D), T))
        xs = Fun{D, Pr, 0}(mf, regular_simplex(Val(D), T))
        geom = Geometry(mf, dom, xs)

        for R in 0:D
            @test length(geom.volumes[R].values) == binomial(D+1, R+1)
        end
        # <https://en.wikipedia.org/wiki/Simplex#Volume>
        @test geom.volumes[D].values[1] ≈ sqrt(T(D+1)/2^D) / factorial(D)
    end



    # Cubes are not Delaunay
    # @testset "Cube" begin
    #     mf = hypercube_manifold(Val(D))
    #     xs = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #         mf,
    #         [Chain{V,1}(SVector{D}(T((n-1) & (1<<(d-1)) != 0) for d in 1:D))
    #          for n in 1:1<<D])
    #     geom = Geometry(mf, dom, xs)
    # 
    #     @test length(geom.volumes[0].values) == 2^D
    #     @test length(geom.volumes[D].values) == factorial(D)
    #     @test sum(geom.volumes[D].values) ≈ 1
    # end
end



@testset "Geometry ops D=$D P=$P R=$R" for D in 1:Dmax, P in (Pr, Dl), R in 0:D
    S = Signature(D)
    V = SubManifold(S)
    T = Float64

    dom = Domain(Chain{V,1}(zeros(SVector{D,T})),
                 Chain{V,1}(ones(SVector{D,T})))

    # Empty manifold
    mf0 = DManifold(Val(D))
    xs0 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(mf0, fulltype(Chain{V,1,T})[])
    geom0 = Geometry(mf0, dom, xs0)

    # Not Delaunay
    # # Simplex
    # mf1 = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    # xs1 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #     mf1,
    #     [Chain{V,1}(SVector{D}(T(d+1 == n) for d in 1:D)) for n in 1:D+1])
    # geom1 = Geometry(mf1, dom, xs1)

    # Regular simplex
    mf2 = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    xs2 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(mf2, regular_simplex(Val(D), T))
    geom2 = Geometry(mf2, dom, xs2)

    # Not Delaunay
    # # Cube
    # mf3 = hypercube_manifold(Val(D))
    # xs3 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #     mf3,
    #     [Chain{V,1}(SVector{D}(T((n-1) & (1<<(d-1)) != 0) for d in 1:D))
    #      for n in 1:1<<D])
    # geom3 = Geometry(mf3, dom, xs3)

    # geoms = [geom0, geom1, geom2, geom3]
    geoms = [geom0, geom2]

    for geom in geoms

        f0 = ones(Fun{D, P, R, T}, geom.mf)
        f1 = id(Fun{D, P, R, T}, geom.mf)
        fs = [f0, f1]

        for f in fs
            h = hodge(Val(P), Val(R), geom)
            @test all(>(0), h.values.diag)
            hf = h*f
            hf::Fun{D, !P, R, T}
        end

        if P == Pr
            if R > 0
                c = coderiv(Val(P), Val(R), geom)
                for f in fs
                    cf = c*f
                    cf::Fun{D, P, R-1, T}
                end
            end

            for f in fs
                l = laplace(Val(P), Val(R), geom)
                lf = l*f
                lf::Fun{D, P, R, T}
            end
        end
    end
end
