using DDF

using LinearAlgebra

@testset "Empty manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = empty_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == 0
    end
    @test lookup(mfd, 0, D) ≡ mfd.simplices[D]
    for R in 0:D
        @test isempty(volumes(mfd, R))
        @test isempty(dualvolumes(mfd, R))
    end
end

@testset "Simplex manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = simplex_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == binomial(D + 1, R + 1)
    end
    @test lookup(mfd, 0, D) ≡ mfd.simplices[D]

    N = D + 1
    for i in 1:N, j in (i + 1):N
        @test norm(coords(mfd)[i] - coords(mfd)[j]) ≈ 1
    end

    for R in 0:D
        # <https://en.wikipedia.org/wiki/Simplex#Volume>
        vol = sqrt(S(R + 1) / 2^R) / factorial(R)
        @test all(≈(vol), volumes(mfd, R))
        @test minimum(volumes(mfd, R)) / maximum(volumes(mfd, R)) ≥ 0.01
        @test all(>(0), dualvolumes(mfd, R))
        @test minimum(dualvolumes(mfd, R)) / maximum(dualvolumes(mfd, R)) ≥ 0.01
    end
    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    vol = sqrt(S(D + 1) / 2^D) / factorial(D)
    @test all(==(1), volumes(mfd, 0))
    @test sum(volumes(mfd, D)) ≈ vol
    @test sum(dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), dualvolumes(mfd, D))
end

@testset "Orthogonal simplex manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = orthogonal_simplex_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == binomial(D + 1, R + 1)
    end
    @test lookup(mfd, 0, D) ≡ mfd.simplices[D]
    for R in 0:D
        @test minimum(volumes(mfd, R)) / maximum(volumes(mfd, R)) ≥ 0.01
        if mfd.dualkind == BarycentricDuals || (D ≤ 2 &&
            mfd.dualkind == CircumcentricDuals &&
            mfd.use_weighted_duals)
            @test all(>(0), dualvolumes(mfd, R))
            @test minimum(dualvolumes(mfd, R)) / maximum(dualvolumes(mfd, R)) ≥
                  0.01
        end
    end

    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    vol = S(1) / factorial(D)
    @test all(==(1), volumes(mfd, 0))
    @test sum(volumes(mfd, D)) ≈ vol
    @test sum(dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), dualvolumes(mfd, D))
end

@testset "Hypercube manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = hypercube_manifold(Val(D), S)
    @test invariant(mfd)
    @test nsimplices(mfd, 0) == 2^D
    @test nsimplices(mfd, D) == factorial(D)
    # TODO: Find rule for other dimensions
    @test lookup(mfd, 0, D) ≡ mfd.simplices[D]
    for R in 0:D
        @test minimum(volumes(mfd, R)) / maximum(volumes(mfd, R)) ≥ 0.01
        if mfd.dualkind == BarycentricDuals ||
           (mfd.dualkind == CircumcentricDuals && mfd.use_weighted_duals)
            @test all(>(0), dualvolumes(mfd, R))
            @test minimum(dualvolumes(mfd, R)) / maximum(dualvolumes(mfd, R)) ≥
                  0.01
        end
    end

    vol = 1
    @test all(==(1), volumes(mfd, 0))
    @test sum(volumes(mfd, D)) ≈ vol
    @test sum(dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), dualvolumes(mfd, D))
end

@testset "Delaunay hypercube manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = delaunay_hypercube_manifold(Val(D), S)
    @test invariant(mfd)
    @test nsimplices(mfd, 0) == 2^D
    @test lookup(mfd, 0, D) ≡ mfd.simplices[D]
    for R in 0:D
        @test minimum(volumes(mfd, R)) / maximum(volumes(mfd, R)) ≥ 0.01
        if mfd.dualkind == BarycentricDuals || (D ≤ 3 &&
            mfd.dualkind == CircumcentricDuals &&
            mfd.use_weighted_duals)
            @test all(>(0), dualvolumes(mfd, R))
            @test minimum(dualvolumes(mfd, R)) / maximum(dualvolumes(mfd, R)) ≥
                  0.01
        end
    end

    vol = 1
    @test all(==(1), volumes(mfd, 0))
    @test sum(volumes(mfd, D)) ≈ vol
    @test sum(dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), dualvolumes(mfd, D))
end

@testset "Large delaunay hypercube manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = large_delaunay_hypercube_manifold(Val(D), S)
    @test invariant(mfd)
    n = D == 0 ? 1 : round(Int, nsimplices(mfd, 0)^(1 / D)) - 1
    @test nsimplices(mfd, 0) == (n + 1)^D
    @test lookup(mfd, 0, D) ≡ mfd.simplices[D]
    for R in 0:D
        if mfd.dualkind == BarycentricDuals || D ≤ 3
            @test minimum(volumes(mfd, R)) / maximum(volumes(mfd, R)) ≥ 0.01
        end
        if mfd.dualkind == BarycentricDuals || (D ≤ 2 &&
            mfd.dualkind == CircumcentricDuals &&
            mfd.use_weighted_duals)
            @test all(>(0), dualvolumes(mfd, R))
            @test minimum(dualvolumes(mfd, R)) / maximum(dualvolumes(mfd, R)) ≥
                  0.01
        end
    end

    vol = 1
    @test all(==(1), volumes(mfd, 0))
    @test sum(volumes(mfd, D)) ≈ vol
    @test sum(dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), dualvolumes(mfd, D))
end

@testset "Refined simplex manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = refined_simplex_manifold(Val(D), S)
    @test invariant(mfd)
    @test nsimplices(mfd, 0) == binomial(D + 1, 1) + binomial(D + 1, 2)
    @test nsimplices(mfd, D) == 2^D
    @test lookup(mfd, 0, D) ≡ mfd.simplices[D]

    for R in 0:D
        if R == 0 || R == D
            # <https://en.wikipedia.org/wiki/Simplex#Volume>
            vol = sqrt(S(R + 1) / 2^R) / factorial(R) / 2^R
            @test all(≈(vol), volumes(mfd, R))
        end
        if mfd.dualkind == BarycentricDuals || D ≤ 3
            @test minimum(volumes(mfd, R)) / maximum(volumes(mfd, R)) ≥ 0.01
        end
        if mfd.dualkind == BarycentricDuals || (D ≤ 4 &&
            mfd.dualkind == CircumcentricDuals &&
            mfd.use_weighted_duals)
            @test all(>(0), dualvolumes(mfd, R))
            if D ≤ 3
                @test minimum(dualvolumes(mfd, R)) /
                      maximum(dualvolumes(mfd, R)) ≥ 0.01
            end
        end
    end
    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    vol = sqrt(S(D + 1) / 2^D) / factorial(D)
    @test all(==(1), volumes(mfd, 0))
    @test sum(volumes(mfd, D)) ≈ vol
    @test sum(dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), dualvolumes(mfd, D))
end

@testset "Boundary simplex manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = boundary_simplex_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == binomial(D + 2, R + 1)
    end

    N = D + 2
    for i in 1:N, j in (i + 1):N
        @test norm(coords(mfd)[i] - coords(mfd)[j]) ≈ 1
    end

    for R in 0:D
        # <https://en.wikipedia.org/wiki/Simplex#Volume>
        vol = sqrt(S(R + 1) / 2^R) / factorial(R)
        @test all(≈(vol), volumes(mfd, R))
        @test minimum(volumes(mfd, R)) / maximum(volumes(mfd, R)) ≥ 0.01
        @test all(>(0), dualvolumes(mfd, R))
        @test minimum(dualvolumes(mfd, R)) / maximum(dualvolumes(mfd, R)) ≥ 0.01
    end
end
