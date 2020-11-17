using DDF

using LinearAlgebra
using StaticArrays

@testset "Empty manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = empty_manifold(Val(D), S; optimize_mesh=false)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == 0
    end
    @test get_lookup(mfd, 0, D) ≡ get_simplices(mfd, D)
    for R in 0:D
        @test isempty(get_volumes(mfd, R))
        @test isempty(get_dualvolumes(mfd, R))
    end
end

@testset "Simplex manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = simplex_manifold(Val(D), S; optimize_mesh=false)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == binomial(D + 1, R + 1)
    end
    @test get_lookup(mfd, 0, D) ≡ get_simplices(mfd, D)

    N = D + 1
    for i in ID{0}(1):ID{0}(N), j in (i + 1):ID{0}(N)
        @test norm(get_coords(mfd)[i] - get_coords(mfd)[j]) ≈ 1
    end

    for R in 0:D
        # <https://en.wikipedia.org/wiki/Simplex#Volume>
        vol = sqrt(S(R + 1) / 2^R) / factorial(R)
        @test all(≈(vol), get_volumes(mfd, R))
        @test minimum(get_volumes(mfd, R)) / maximum(get_volumes(mfd, R)) ≥ 0.01
        @test all(>(0), get_dualvolumes(mfd, R))
        @test minimum(get_dualvolumes(mfd, R)) /
              maximum(get_dualvolumes(mfd, R)) ≥ 0.01
    end
    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    vol = sqrt(S(D + 1) / 2^D) / factorial(D)
    @test all(==(1), get_volumes(mfd, 0))
    @test sum(get_volumes(mfd, D)) ≈ vol
    @test sum(get_dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), get_dualvolumes(mfd, D))
end

@testset "Orthogonal simplex manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = orthogonal_simplex_manifold(Val(D), S; optimize_mesh=false)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == binomial(D + 1, R + 1)
    end
    @test get_lookup(mfd, 0, D) ≡ get_simplices(mfd, D)
    for R in 0:D
        @test minimum(get_volumes(mfd, R)) / maximum(get_volumes(mfd, R)) ≥ 0.01
        if mfd.dualkind == BarycentricDuals || (D ≤ 2 &&
            mfd.dualkind == CircumcentricDuals &&
            mfd.use_weighted_duals)
            @test all(>(0), get_dualvolumes(mfd, R))
            @test minimum(get_dualvolumes(mfd, R)) /
                  maximum(get_dualvolumes(mfd, R)) ≥ 0.01
        end
    end

    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    vol = S(1) / factorial(D)
    @test all(==(1), get_volumes(mfd, 0))
    @test sum(get_volumes(mfd, D)) ≈ vol
    @test sum(get_dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), get_dualvolumes(mfd, D))
end

@testset "Hypercube manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = hypercube_manifold(Val(D), S)
    @test invariant(mfd)
    @test nsimplices(mfd, 0) == 2^D
    @test nsimplices(mfd, D) == factorial(D)
    # TODO: Find rule for other dimensions
    @test get_lookup(mfd, 0, D) ≡ get_simplices(mfd, D)
    for R in 0:D
        @test minimum(get_volumes(mfd, R)) / maximum(get_volumes(mfd, R)) ≥ 0.01
        if mfd.dualkind == BarycentricDuals ||
           (mfd.dualkind == CircumcentricDuals && mfd.use_weighted_duals)
            @test all(>(0), get_dualvolumes(mfd, R))
            @test minimum(get_dualvolumes(mfd, R)) /
                  maximum(get_dualvolumes(mfd, R)) ≥ 0.01
        end
    end

    vol = 1
    @test all(==(1), get_volumes(mfd, 0))
    @test sum(get_volumes(mfd, D)) ≈ vol
    @test sum(get_dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), get_dualvolumes(mfd, D))
end

@testset "Large hypercube manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = large_hypercube_manifold(Val(D), S; nelts=ntuple(d -> 2, D),
                                   optimize_mesh=false)
    @test invariant(mfd)
    @test nsimplices(mfd, 0) == 3^D
    @test nsimplices(mfd, D) == 2^D * factorial(D)
    # TODO: Find rule for other dimensions
    @test get_lookup(mfd, 0, D) ≡ get_simplices(mfd, D)
    for R in 0:D
        @test minimum(get_volumes(mfd, R)) / maximum(get_volumes(mfd, R)) ≥ 0.01
        if mfd.dualkind == BarycentricDuals ||
           (mfd.dualkind == CircumcentricDuals && mfd.use_weighted_duals)
            @test all(>(0), get_dualvolumes(mfd, R))
            @test minimum(get_dualvolumes(mfd, R)) /
                  maximum(get_dualvolumes(mfd, R)) ≥ (D ≤ 3 ? 0.01 : 0.001)
        end
    end

    vol = 1
    @test all(==(1), get_volumes(mfd, 0))
    @test sum(get_volumes(mfd, D)) ≈ vol
    if !(sum(get_dualvolumes(mfd, 0)) ≈ vol)
        @show D extrema(get_dualvolumes(mfd, 0))
    end
    @test sum(get_dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), get_dualvolumes(mfd, D))
end

# @testset "Large hypertorus manifold D=$D" for D in 0:Dmax
#     S = Float64
#     mfd = large_hypercube_manifold(Val(D), S; nelts=ntuple(d -> 2, D),
#                                    torus=true)
#     @test invariant(mfd)
#     @test nsimplices(mfd, 0) == 2^D
#     @test nsimplices(mfd, D) == 2^D * factorial(D)
#     # TODO: Find rule for other dimensions
#     @test get_lookup(mfd, 0, D) ≡ get_simplices(mfd, D)
#     for R in 0:D
#         @test minimum(get_volumes(mfd, R)) / maximum(get_volumes(mfd, R)) ≥ 0.01
#         if mfd.dualkind == BarycentricDuals ||
#            (mfd.dualkind == CircumcentricDuals && mfd.use_weighted_duals)
#             @test all(>(0), get_dualvolumes(mfd, R))
#             @test minimum(get_dualvolumes(mfd, R)) /
#                   maximum(get_dualvolumes(mfd, R)) ≥ 0.01
#         end
#     end
# 
#     vol = 1
#     @test all(==(1), get_volumes(mfd, 0))
#     @test sum(get_volumes(mfd, D)) ≈ vol
#     @test sum(get_dualvolumes(mfd, 0)) ≈ vol
#     @test all(==(1), get_dualvolumes(mfd, D))
# end

@testset "Delaunay hypercube manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = delaunay_hypercube_manifold(Val(D), S)
    @test invariant(mfd)
    @test nsimplices(mfd, 0) == 2^D
    @test get_lookup(mfd, 0, D) ≡ get_simplices(mfd, D)
    for R in 0:D
        @test minimum(get_volumes(mfd, R)) / maximum(get_volumes(mfd, R)) ≥ 0.01
        if mfd.dualkind == BarycentricDuals || (D ≤ 3 &&
            mfd.dualkind == CircumcentricDuals &&
            mfd.use_weighted_duals)
            @test all(>(0), get_dualvolumes(mfd, R))
            @test minimum(get_dualvolumes(mfd, R)) /
                  maximum(get_dualvolumes(mfd, R)) ≥ 0.01
        end
    end

    vol = 1
    @test all(==(1), get_volumes(mfd, 0))
    @test sum(get_volumes(mfd, D)) ≈ vol
    @test sum(get_dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), get_dualvolumes(mfd, D))
end

@testset "Large delaunay hypercube manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = large_delaunay_hypercube_manifold(Val(D), S)
    @test invariant(mfd)
    n = D == 0 ? 1 : round(Int, nsimplices(mfd, 0)^(1 / D)) - 1
    @test nsimplices(mfd, 0) == (n + 1)^D
    @test get_lookup(mfd, 0, D) ≡ get_simplices(mfd, D)
    for R in 0:D
        if mfd.dualkind == BarycentricDuals || D ≤ 3
            @test minimum(get_volumes(mfd, R)) / maximum(get_volumes(mfd, R)) ≥
                  0.01
        end
        if mfd.dualkind == BarycentricDuals || (D ≤ 2 &&
            mfd.dualkind == CircumcentricDuals &&
            mfd.use_weighted_duals)
            @test all(>(0), get_dualvolumes(mfd, R))
            @test minimum(get_dualvolumes(mfd, R)) /
                  maximum(get_dualvolumes(mfd, R)) ≥ 0.01
        end
    end

    vol = 1
    @test all(==(1), get_volumes(mfd, 0))
    @test sum(get_volumes(mfd, D)) ≈ vol
    @test sum(get_dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), get_dualvolumes(mfd, D))
end

# @testset "Random hypercube manifold D=$D" for D in 0:Dmax
#     S = Float64
#     xmin = SVector{D,S}(0 for d in 1:D)
#     xmax = SVector{D,S}(1 for d in 1:D)
#     n = 4
#     mfd = random_hypercube_manifold(xmin, xmax, n)
#     @test invariant(mfd)
#     @test nsimplices(mfd, 0) == (n + 2)^D
#     @test get_lookup(mfd, 0, D) ≡ get_simplices(mfd, D)
#     for R in 0:D
#         @test minimum(get_volumes(mfd, R)) / maximum(get_volumes(mfd, R)) ≥ 0.01
#         if mfd.dualkind == BarycentricDuals ||
#            (mfd.dualkind == CircumcentricDuals && mfd.use_weighted_duals)
#             @test all(>(0), get_dualvolumes(mfd, R))
#             @test minimum(get_dualvolumes(mfd, R)) /
#                   maximum(get_dualvolumes(mfd, R)) ≥ 0.01
#         end
#     end
# 
#     vol = 1
#     @test all(==(1), get_volumes(mfd, 0))
#     @test sum(get_volumes(mfd, D)) ≈ vol
#     if !(sum(get_dualvolumes(mfd, 0)) ≈ vol)
#         @show extrema(get_dualvolumes(mfd, 0))
#     end
#     @test sum(get_dualvolumes(mfd, 0)) ≈ vol
#     @test all(==(1), get_dualvolumes(mfd, D))
# end

@testset "Refined simplex manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = refined_simplex_manifold(Val(D), S)
    @test invariant(mfd)
    @test nsimplices(mfd, 0) == binomial(D + 1, 1) + binomial(D + 1, 2)
    @test nsimplices(mfd, D) == 2^D
    @test get_lookup(mfd, 0, D) ≡ get_simplices(mfd, D)

    for R in 0:D
        if R == 0 || R == D
            # <https://en.wikipedia.org/wiki/Simplex#Volume>
            vol = sqrt(S(R + 1) / 2^R) / (factorial(R) * 2^R)
            @test all(≈(vol), get_volumes(mfd, R))
        end
        if mfd.dualkind == BarycentricDuals || D ≤ 3
            @test minimum(get_volumes(mfd, R)) / maximum(get_volumes(mfd, R)) ≥
                  0.01
        end
        if mfd.dualkind == BarycentricDuals || (D ≤ 4 &&
            mfd.dualkind == CircumcentricDuals &&
            mfd.use_weighted_duals)
            @test all(>(0), get_dualvolumes(mfd, R))
            if D ≤ 3
                @test minimum(get_dualvolumes(mfd, R)) /
                      maximum(get_dualvolumes(mfd, R)) ≥ 0.01
            end
        end
    end
    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    vol = sqrt(S(D + 1) / 2^D) / factorial(D)
    @test all(==(1), get_volumes(mfd, 0))
    @test sum(get_volumes(mfd, D)) ≈ vol
    @test sum(get_dualvolumes(mfd, 0)) ≈ vol
    @test all(==(1), get_dualvolumes(mfd, D))
end

@testset "Boundary simplex manifold D=$D" for D in 0:Dmax
    S = Float64
    mfd = boundary_simplex_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == binomial(D + 2, R + 1)
    end

    N = D + 2
    for i in ID{0}(1):ID{0}(N), j in (i + 1):ID{0}(N)
        @test norm(get_coords(mfd)[i] - get_coords(mfd)[j]) ≈ 1
    end

    for R in 0:D
        # <https://en.wikipedia.org/wiki/Simplex#Volume>
        vol = sqrt(S(R + 1) / 2^R) / factorial(R)
        @test all(≈(vol), get_volumes(mfd, R))
        @test minimum(get_volumes(mfd, R)) / maximum(get_volumes(mfd, R)) ≥ 0.01
        @test all(>(0), get_dualvolumes(mfd, R))
        @test minimum(get_dualvolumes(mfd, R)) /
              maximum(get_dualvolumes(mfd, R)) ≥ 0.01
    end
end
