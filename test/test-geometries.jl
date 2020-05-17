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
    # # Rectilinear pyramid
    # mf = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    # xs = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #     mf,
    #     fulltype(Chain{V,1,T})[
    #         Chain{V,1}(sarray(T, d -> T(d+1 == n), Val(D))) for n in 1:D+1])
    # Regular simplex
    mf = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    xs = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(mf, regular_simplex(Val(D), T))
    geom = Geometry(mf, dom, xs)
    # if D==2 || D==3
    #     @show geom.mf.nvertices
    #     @show geom.mf.simplices
    #     for d in 1:D
    #         @show d geom.mf.boundaries[d]
    #     end
    #     @show geom.dom
    #     @show geom.coords.values
    #     for d in 1:D
    #         @show d geom.volumes[d].values
    #     end
    #     @show geom.dualcoords.values
    # end
    for R in 1:D
        @test length(geom.volumes[R].values) == binomial(D+1, R+1)
    end
    @test length(geom.volumes[D].values) == 1
    # @test geom.volumes[D].values[1] ≈ one(T)/factorial(D)
    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    @test geom.volumes[D].values[1] ≈ sqrt(T(D+1)/2^D) / factorial(D)
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

    # # Simplex
    # mf1 = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    # xs1 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #     mf1,
    #     [Chain{V,1}(sarray(T, d -> T(d+1 == n), Val(D))) for n in 1:D+1])
    # geom1 = Geometry(mf1, dom, xs1)

    # Regular simplex
    mf1 = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    xs1 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(mf1, regular_simplex(Val(D), T))
    geom1 = Geometry(mf1, dom, xs1)

    # # Cube
    # mf2 = hypercube_manifold(Val(D))
    # xs2 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #     mf2,
    #     [Chain{V,1}(sarray(T, d -> T((n-1) & (1<<(d-1)) != 0), Val(D)))
    #      for n in 1:1<<D])
    # geom2 = Geometry(mf2, dom, xs2)

    geoms = [geom0, geom1]

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
