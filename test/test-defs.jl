using DDF

using ComputedFieldTypes
using Grassmann
using Test



@testset "bitsign" begin
    for b in false:true
        @test abs(bitsign(b)) === 1
        @test signbit(bitsign(b)) === b
        @test bitsign(Int(b)) === bitsign(b)
    end
    for n in 1:100
        i = Int(rand(Int8))
        j = Int(rand(Int8))
        @test bitsign(i) * bitsign(j) == bitsign(i + j)
    end
end



@testset "Projective Geometric Algebra D=$D" for D in 1:Dmax
    S = Signature(D)
    V = SubManifold(S)
    @test iseuclidean(V)
    T = Rational{Int128}
    for n in 1:100
        x = rand(Chain{V,1,T})
        x::Chain
        @test iseuclidean(x)
        px = projective(x)
        px::Chain
        @test isprojective(px)
        @test euclidean(px) == x
        # TODO: add more checks
    end
end



@testset "Conformal Geometric Algebra D=$D" for D in 1:Dmax
    S = Signature(D)
    V = SubManifold(S)
    @test iseuclidean(V)
    T = Rational{Int128}
    for n in 1:100
        x = rand(Chain{V,1,T})
        x::Chain
        @test iseuclidean(x)
        cx = conformal(x)
        cx::Chain
        @test isconformal(cx)
        @test euclidean(cx) == x
        # TODO: add more checks
    end
end



@testset "Circumcentre D=$D" for D in 1:Dmax
    S = Signature(D)
    V = SubManifold(S)
    # T = Rational{Int128}
    T = Rational{BigInt}
    for R in 1:D+1
        xs = sarray(fulltype(Chain{V,1,T}), r -> rand(Chain{V,1,T}), Val(R))
        cc = circumcentre(xs)
        cc::Chain{V,1,T}
        # Check that circumcentre has same distance from all vertices
        r2 = scalar(abs2(xs[1] - cc)).v
        @test all(==(r2), scalar(abs2(xs[i] - cc)).v for i in 1:R)
        # Check that circumcentre lies in hyperplane spanned by
        # vertices
        if length(xs)==1
            subspace = xs[1]
        else
            subspace = ∧(xs)
        end
        @test cc ∧ subspace == 0
    end
end



@testset "Regular simplex D=$D" for D in 1:Dmax
    N = D+1
    T = Float64
    s = regular_simplex(Val(D), T)
    for i in 1:N, j in i+1:N
        @test scalar(abs2(s[i] - s[j])).v ≈ 1
    end
end
