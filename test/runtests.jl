using Test
using DDF

@testset "Domain D=$D" for D in 0:2
    dom0 = empty_domain(Val(D))
    for R in 0:D
        @test dim(Val(R), dom0) == 0
    end

    dom1 = cell_domain(Tuple(1:D+1))
    if D == 0
        @test dim(Val(0), dom1) == 1
    elseif D == 1
        @test dim(Val(0), dom1) == 2
        @test dim(Val(1), dom1) == 1
    elseif D == 2
        @test dim(Val(0), dom1) == 3
        @test dim(Val(1), dom1) == 3
        @test dim(Val(2), dom1) == 1
    else
        @assert false
    end
end

# Random rationals
Base.rand(::Type{Rational{T}}) where {T} =
    Rational{T}(rand(-T(1000):T(1000))) / T(1000)
Base.rand(::Type{Rational{T}}, n::Int) where {T} =
    Rational{T}[rand(Rational{T}) for i in 1:n]

# Random functions
Base.rand(::Type{Fun{D, R, T}}, dom::Domain{D}) where {D, R, T} =
    Fun{D, R, T}(dom, rand(T, dim(Val(R), dom)))

@testset "Fun D=$D R=$R" for D in 0:2, R in 0:D
    dom = cell_domain(Tuple(1:D+1))

    T = Rational{Int64}
    z = zero(Fun{D, R, T}, dom)
    e = one(Fun{D, R, T}, dom)
    i = id(Fun{D, R, T}, dom)
    f = rand(Fun{D, R, T}, dom)
    g = rand(Fun{D, R, T}, dom)
    h = rand(Fun{D, R, T}, dom)
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
