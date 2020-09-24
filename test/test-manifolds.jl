using DDF

using LinearAlgebra

@testset "Empty manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = empty_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == 0
    end
    @test mfd.lookup[(0, D)] === mfd.simplices[D]
    for R in 0:D
        @test all(>(0), mfd.dualvolumes[R])
    end

    for R in 0:D
        @test isempty(mfd.volumes[R])
        @test isempty(mfd.dualvolumes[R])
    end
end

@testset "Simplex manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = simplex_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == binomial(D + 1, R + 1)
    end
    @test mfd.lookup[(0, D)] === mfd.simplices[D]

    N = D + 1
    for i in 1:N, j in (i + 1):N
        @test norm(mfd.coords[0][i] - mfd.coords[0][j]) ≈ 1
    end

    for R in 0:D
        # <https://en.wikipedia.org/wiki/Simplex#Volume>
        vol = sqrt(S(R + 1) / 2^R) / factorial(R)
        @test all(≈(vol), mfd.volumes[R])
        @test all(>(0), mfd.dualvolumes[R])
    end
    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    vol = sqrt(S(D + 1) / 2^D) / factorial(D)
    @test all(==(1), mfd.volumes[0])
    @test sum(mfd.volumes[D]) ≈ vol
    @test sum(mfd.dualvolumes[0]) ≈ vol
    @test all(==(1), mfd.dualvolumes[D])
end

@testset "Orthogonal simplex manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = orthogonal_simplex_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == binomial(D + 1, R + 1)
    end
    @test mfd.lookup[(0, D)] === mfd.simplices[D]
    for R in 0:D
        @test all(>(0), mfd.dualvolumes[R])
    end

    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    vol = S(1) / factorial(D)
    @test all(==(1), mfd.volumes[0])
    @test sum(mfd.volumes[D]) ≈ vol
    @test sum(mfd.dualvolumes[0]) ≈ vol
    @test all(==(1), mfd.dualvolumes[D])
end

@testset "Hypercube manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = hypercube_manifold(Val(D), S)
    @test invariant(mfd)
    @test nsimplices(mfd, 0) == 2^D
    @test nsimplices(mfd, D) == factorial(D)
    # TODO: Find rule for other dimensions
    @test mfd.lookup[(0, D)] === mfd.simplices[D]
    for R in 0:D
        @test all(>(0), mfd.dualvolumes[R])
    end

    vol = 1
    @test all(==(1), mfd.volumes[0])
    @test sum(mfd.volumes[D]) ≈ vol
    @test sum(mfd.dualvolumes[0]) ≈ vol
    @test all(==(1), mfd.dualvolumes[D])
end

# Delaunay hypercube manifolds are only completely well-centred for `D
# <= 4`
@testset "Delaunay hypercube manifolds D=$D" for D in 0:min(Dmax, 4)
    S = Float64
    mfd = delaunay_hypercube_manifold(Val(D), S)
    @test invariant(mfd)
    @test nsimplices(mfd, 0) == 2^D
    @test mfd.lookup[(0, D)] === mfd.simplices[D]
    for R in 0:D
        @test all(>(0), mfd.dualvolumes[R])
    end

    vol = 1
    @test all(==(1), mfd.volumes[0])
    @test sum(mfd.volumes[D]) ≈ vol
    @test sum(mfd.dualvolumes[0]) ≈ vol
    @test all(==(1), mfd.dualvolumes[D])
end

# Large Delaunay hypercube manifolds are only completely well-centred
# for `D <= 1`
@testset "Large delaunay hypercube manifolds D=$D" for D in 0:min(1, Dmax)
    S = Float64
    mfd = large_delaunay_hypercube_manifold(Val(D), S)
    @test invariant(mfd)
    n = D == 0 ? 1 : round(Int, nsimplices(mfd, 0)^(1 / D)) - 1
    @test nsimplices(mfd, 0) == (n + 1)^D
    @test mfd.lookup[(0, D)] === mfd.simplices[D]
    for R in 0:D
        @test all(>(0), mfd.dualvolumes[R])
    end

    vol = 1
    @test all(==(1), mfd.volumes[0])
    @test sum(mfd.volumes[D]) ≈ vol
    @test sum(mfd.dualvolumes[0]) ≈ vol
    @test all(==(1), mfd.dualvolumes[D])
end

# Refined simplex manifolds are only completely well-centred for `D <= 3`
@testset "Refined simplex manifold D=$D" for D in 0:min(3, Dmax)
    S = Float64
    mfd = refined_simplex_manifold(Val(D), S)
    @test invariant(mfd)
    @test nsimplices(mfd, 0) == binomial(D + 1, 1) + binomial(D + 1, 2)
    @test nsimplices(mfd, D) == 2^D
    @test mfd.lookup[(0, D)] === mfd.simplices[D]

    for R in 0:D
        if R == 0 || R == D
            # <https://en.wikipedia.org/wiki/Simplex#Volume>
            vol = sqrt(S(R + 1) / 2^R) / factorial(R) / 2^R
            @test all(≈(vol), mfd.volumes[R])
        end
        @test all(>(0), mfd.dualvolumes[R])
    end
    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    vol = sqrt(S(D + 1) / 2^D) / factorial(D)
    @test all(==(1), mfd.volumes[0])
    @test sum(mfd.volumes[D]) ≈ vol
    @test sum(mfd.dualvolumes[0]) ≈ vol
    @test all(==(1), mfd.dualvolumes[D])
end
