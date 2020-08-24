using DDF

@testset "empty Manifolds D=$D" for D in 0:Dmax
    T = Rational{Int64}
    mfd = empty_manifold(Val(D), T)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == 0
    end
end

@testset "simplex Manifolds D=$D" for D in 0:Dmax
    T = Float64
    mfd = simplex_manifold(Val(D), T)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == binomial(D + 1, R + 1)
    end
end

@testset "hypercube Manifolds D=$D" for D in 0:Dmax
    T = Rational{Int64}
    mfd = hypercube_manifold(Val(D), T)
    @test invariant(mfd)
    @test nsimplices(mfd, D) == factorial(D)
    @test nsimplices(mfd, 0) == 2^D
    # TODO: Find rule for other dimensions
end
