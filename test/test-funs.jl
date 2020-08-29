using DDF

using StaticArrays
using Test

@testset "Fun D=$D P=$P R=$R" for D in 0:Dmax, P in (Pr, Dl), R in 0:D
    T = Rational{Int64}
    mfd = hypercube_manifold(Val(D), T)

    z = zero(Fun{D,P,R,D,T,T}, mfd)
    e = id(Fun{D,P,0,D,T,SVector{D,T}}, mfd)
    f = rand(Fun{D,P,R,D,T,T}, mfd)
    g = rand(Fun{D,P,R,D,T,T}, mfd)
    h = rand(Fun{D,P,R,D,T,T}, mfd)
    a = rand(T)
    b = rand(T)

    # Vector space

    @test +f == f
    @test (f + g) + h == f + (g + h)

    @test z + f == f
    @test f + z == f

    @test f + (-f) == z
    @test (-f) + f == z
    @test f - g == f + (-g)

    @test (a * b) * f == a * (b * f)
    @test f * (a * b) == (f * a) * b

    @test one(T) * a == a
    @test one(T) * f == f
    @test f * one(T) == f

    @test zero(T) * a == zero(T)
    @test zero(T) * f == z
    @test f * zero(T) == z

    if a != zero(T)
        @test a * inv(a) == one(T)

        @test inv(a) * (a * f) == f
        @test (f * a) * inv(a) == f

        @test inv(a) * f == a \ f
        @test f * inv(a) == f / a
    end

    @test (a + b) * f == a * f + b * f
    @test f * (a + b) == f * a + f * b

    @test a * (f + g) == a * f + a * g
    @test (f + g) * a == f * a + g * a

    # Nonlinear transformations

    e2 = map(x -> 2x, e)
    @test [2x for x in e] == [x for x in e2]

    @test map(+, f, g) == f + g
end
