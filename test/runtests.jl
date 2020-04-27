using ComputedFieldTypes
using DDF
using LinearAlgebra
using SparseArrays
using StaticArrays
using Test



@testset "Manifold D=$D" for D in 0:4

    function checkboundary2(mf::Manifold{D}) where {D}
        for R in 2:D
            boundary2 = dropzeros(mf.boundaries[R-1] * mf.boundaries[R])
            @test nnz(boundary2) == 0
        end
    end

    mf0 = Manifold(Val(D))
    for R in 0:D
        @test dim(Val(R), mf0) == 0
    end
    checkboundary2(mf0)

    mf1 = Manifold(MSimplex(SVector{D+1}(1:D+1)))
    if D == 0
        @test dim(Val(0), mf1) == 1
    elseif D == 1
        @test dim(Val(0), mf1) == 2
        @test dim(Val(1), mf1) == 1
    elseif D == 2
        @test dim(Val(0), mf1) == 3
        @test dim(Val(1), mf1) == 3
        @test dim(Val(2), mf1) == 1
    elseif D == 3
        @test dim(Val(0), mf1) == 4
        @test dim(Val(1), mf1) == 6
        @test dim(Val(2), mf1) == 4
        @test dim(Val(3), mf1) == 1
    elseif D == 4
        @test dim(Val(0), mf1) == 5
        @test dim(Val(1), mf1) == 10
        @test dim(Val(2), mf1) == 10
        @test dim(Val(3), mf1) == 5
        @test dim(Val(4), mf1) == 1
    else
        @assert false
    end
    checkboundary2(mf1)

    if D == 2
        # a Möbius strip
        mf2 = Manifold(SVector{D+1}.([(1, 2, 4),
                                      (1, 4, 6),
                                      (4, 3, 6),
                                      (6, 3, 5),
                                      (3, 1, 5),
                                      (1, 2, 5)]))
        @test ndims(mf2) == D
        @test dim(Val(0), mf2) == 6
        @test dim(Val(1), mf2) == 12
        @test dim(Val(2), mf2) == 6

        checkboundary2(mf2)
    end

    if D == 1
        # a line
        p0, p2, p1 = 1:3
        mf3 = Manifold(SVector{D+1}.([(p0, p1), (p2, p1)]))
        @test ndims(mf3) == D
        checkboundary2(mf3)
    end

    if D == 2
        # a square
        p00, p02, p20, p22,
        p11 = 1:5
        mf4 = Manifold(SVector{D+1}.([(p00, p02, p11),
                                      (p02, p22, p11),
                                      (p22, p20, p11),
                                      (p20, p00, p11)]))
        @test ndims(mf4) == D
        checkboundary2(mf4)
    end

    if D == 3
        # a cube
        p000, p002, p020, p022, p200, p202, p220, p222, 
        p110, p112, p101, p121, p011, p211, 
        p111 = 1:15
        mf5 = Manifold(SVector{D+1}.([(p000, p020, p110, p111),
                                      (p020, p220, p110, p111),
                                      (p220, p200, p110, p111),
                                      (p200, p000, p110, p111),
                                      (p002, p022, p112, p111),
                                      (p022, p222, p112, p111),
                                      (p222, p202, p112, p111),
                                      (p202, p002, p112, p111),
                                      (p000, p002, p101, p111),
                                      (p002, p202, p101, p111),
                                      (p202, p200, p101, p111),
                                      (p200, p000, p101, p111),
                                      (p020, p022, p121, p111),
                                      (p022, p222, p121, p111),
                                      (p222, p220, p121, p111),
                                      (p220, p020, p121, p111),
                                      (p000, p002, p011, p111),
                                      (p002, p022, p011, p111),
                                      (p022, p020, p011, p111),
                                      (p020, p000, p011, p111),
                                      (p200, p202, p211, p111),
                                      (p202, p222, p211, p111),
                                      (p222, p220, p211, p111),
                                      (p220, p200, p211, p111)]))
        @test ndims(mf5) == D
        checkboundary2(mf5)
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

@testset "Fun D=$D R=$R" for D in 0:4, R in 0:D
    mf = Manifold(MSimplex(SVector{D+1}(1:D+1)))

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



@testset "Geometry D=$D" for D in 0:4
    T = Rational{Int64}
    mf = Manifold(MSimplex(SVector{D+1}(1:D+1)))
    dom = Domain{D, T}(sarray(T, d -> T(0), Val(D)),
                       sarray(T, d -> T(1), Val(D)))
    xs = ntuple(d -> Fun{D, 0, T}(mf, T[d+1 == i for i in 1:D+1]), D)
    cs = Coords{D, T}(mf, dom, xs)
    ccs = circumcentres(cs)
end

@testset "Geometry D=$D" for D in 0:4
    T = Float64
    mf = Manifold(MSimplex(SVector{D+1}(1:D+1)))
    dom = Domain{D, T}(sarray(T, d -> T(0), Val(D)),
                       sarray(T, d -> T(1), Val(D)))
    xs = ntuple(d -> Fun{D, 0, T}(mf, T[d+1 == i for i in 1:D+1]), D)
    cs = Coords{D, T}(mf, dom, xs)
    hodges = hodge(cs)
end
