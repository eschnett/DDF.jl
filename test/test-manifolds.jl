using DDF

using LinearAlgebra

@testset "Empty manifolds D=$D" for D in 0:Dmax
    S = Rational{Int64}
    mfd = empty_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == 0
    end
    if D > 0
        @test mfd.lookup[(0, D)] === mfd.simplices[D]
    end

    for R in 0:D
        @test isempty(mfd.volumes[R])
        @test isempty(mfd.dualvolumes[R])
    end
end

@testset "Simplex manifolds D=$D" for D in 0:Dmax
    S = Float64
    mfd = simplex_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == binomial(D + 1, R + 1)
    end
    if D > 0
        @test mfd.lookup[(0, D)] === mfd.simplices[D]
    end

    N = D + 1
    for i in 1:N, j in (i + 1):N
        @test norm((@view mfd.coords[i, :]) - (@view mfd.coords[j, :])) ≈ 1
    end

    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    vol = sqrt(S(D + 1) / 2^D) / factorial(D)
    @test all(==(1), mfd.volumes[0])
    @test sum(mfd.volumes[D]) ≈ vol
    @test sum(mfd.dualvolumes[0]) ≈ vol
    @test all(==(1), mfd.dualvolumes[D])
end

@testset "Hypercube manifolds D=$D" for D in 0:Dmax
    S = Rational{Int64}
    mfd = hypercube_manifold(Val(D), S)
    @test invariant(mfd)
    @test nsimplices(mfd, D) == factorial(D)
    @test nsimplices(mfd, 0) == 2^D
    # TODO: Find rule for other dimensions
    if D > 0
        @test mfd.lookup[(0, D)] === mfd.simplices[D]
    end

    vol = 1
    @test all(==(1), mfd.volumes[0])
    @test sum(mfd.volumes[D]) ≈ vol
    @test sum(mfd.dualvolumes[0]) ≈ vol
    @test all(==(1), mfd.dualvolumes[D])
end
