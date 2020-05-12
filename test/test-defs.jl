using DDF

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
    end
end
