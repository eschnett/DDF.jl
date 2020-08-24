using DDF

@testset "empty Manifolds D=$D" for D in 0:Dmax
    mempty = empty_manifold(Val(D))
    @test invariant(mempty)
    for R in 0:D
        @test nsimplices(mempty, R) == 0
    end
end

@testset "simplex Manifolds D=$D" for D in 0:Dmax
    msimplex = simplex_manifold(Val(D), Float64)
    @test invariant(msimplex)
    @test nsimplices(msimplex, D) == 1
    # more
    @test nsimplices(msimplex, 0) == D + 1
end
