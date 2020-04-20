using Test
using DDF

@testset "Manifold D=$D" for D in 0:2
    mf0 = empty_manifold(Val(D))
    for R in 0:D
        @test dim(Val(R), mf0) == 0
    end

    mf1 = cell_manifold(Tuple(1:D+1))
    if D == 0
        @test dim(Val(0), mf1) == 1
    elseif D == 1
        @test dim(Val(0), mf1) == 2
        @test dim(Val(1), mf1) == 1
    elseif D == 2
        @test dim(Val(0), mf1) == 3
        @test dim(Val(1), mf1) == 3
        @test dim(Val(2), mf1) == 1
    else
        @assert false
    end

    if D == 2
        # a Möbius strip
        mf2 = simplicial_manifold([(1, 2, 4),
                                   (1, 4, 6),
                                   (4, 3, 6),
                                   (6, 3, 5),
                                   (3, 1, 5),
                                   (1, 2, 5)])
        @test dim(Val(0), mf2) == 6
        @test dim(Val(1), mf2) == 12
        @test dim(Val(2), mf2) == 6
    end
end

# Random rationals
Base.rand(::Type{Rational{T}}) where {T} =
    Rational{T}(rand(-T(1000):T(1000))) / T(1000)
Base.rand(::Type{Rational{T}}, n::Int) where {T} =
    Rational{T}[rand(Rational{T}) for i in 1:n]

# Random functions
Base.rand(::Type{Fun{D, R, T}}, mf::Manifold{D}) where {D, R, T} =
    Fun{D, R, T}(mf, rand(T, dim(Val(R), mf)))

@testset "Fun D=$D R=$R" for D in 0:2, R in 0:D
    mf = cell_manifold(Tuple(1:D+1))

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

    @test T(1) * a == a
    @test T(1) * f == f
    @test f * T(1) == f

    @test T(0) * a == T(0)
    @test T(0) * f == z
    @test f * T(0) == z

    if a != T(0)
        @test a * inv(a) == T(1)

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
