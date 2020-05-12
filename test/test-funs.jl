using DDF

# using ComputedFieldTypes
# using Grassmann
# using LinearAlgebra
# using SparseArrays
# using StaticArrays
using Test



@testset "Fun D=$D R=$R" for D in 0:Dmax, R in 0:D
    mf = DManifold(DSimplex(SVector{D+1}(1:D+1)))

    T = Rational{Int64}
    z = zero(Fun{D, R, T}, mf)
    e = one(Fun{D, R, T}, mf)
    i = id(Fun{D, R, T}, mf)
    f = rand(Fun{D, R, T}, mf)
    g = rand(Fun{D, R, T}, mf)
    h = rand(Fun{D, R, T}, mf)
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

    i2 = map(x->x^2, i)
    @test [x^2 for x in i] == [x for x in i2]

    @test map(+, f, g) == f + g
end
