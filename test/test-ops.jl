using DDF

using LinearAlgebra
using StaticArrays
using Test



# Dmax-1 since these tests are expensive
@testset "Op D=$D P2=$P2 R2=$R2 P1=$P1 R1=$R1" for D in 0:Dmax-1,
    P2 in (Pr, Dl), R2 in 0:D, P1 in (Pr, Dl), R1 in 0:D
    mf = DManifold(DSimplex(SVector{D+1}(1:D+1)))

    T = Rational{Int64}
    z = zero(Op{D, P2, R2, P1, R1, T}, mf)
    e = one(Op{D, P1, R1, P1, R1, T}, mf)
    e2 = one(Op{D, P2, R2, P2, R2, T}, mf)
    A = rand(Op{D, P2, R2, P1, R1, T}, mf)
    B = rand(Op{D, P2, R2, P1, R1, T}, mf)
    C = rand(Op{D, P2, R2, P1, R1, T}, mf)
    f = rand(Fun{D, P1, R1, T}, mf)
    g = rand(Fun{D, P1, R1, T}, mf)
    a = rand(T)
    b = rand(T)

    # Vector space

    @test +A == A
    @test (A + B) + C == A + (B + C)

    @test z + A == A
    @test A + z == A

    @test A + (-A) == z
    @test (-A) + A == z
    @test A - B == A + (-B)

    @test (a * b) * A == a * (b * A)
    @test A * (a * b) == (A * a) * b

    @test one(T) * a == a
    @test one(T) * A == A
    @test A * one(T) == A

    @test zero(T) * a == zero(T)
    @test zero(T) * A == z
    @test A * zero(T) == z

    if a != zero(T)
        @test a * inv(a) == one(T)

        @test inv(a) * (a * A) == A
        @test (A * a) * inv(a) == A

        @test inv(a) * A == a \ A
        @test A * inv(a) == A / a
    end

    @test (a + b) * A == a * A + b * A
    @test A * (a + b) == A * a + A * b

    @test a * (A + B) == a * A + a * B
    @test (A + B) * a == A * a + B * a

    # Category ("Ring")

    @test e2 * A == A
    @test A * e == A

    for P3 in (Pr, Dl), R3 in 0:D, P4 in (Pr, Dl), R4 in 0:D
        F = rand(Op{D, P3, R3, P2, R2, T}, mf)
        G = rand(Op{D, P4, R4, P3, R3, T}, mf)

        @test (G * F) * A == G * (F * A)
    end

    # # Groupoid ("Division ring")
    # 
    # if A != z
    #     @test A * inv(A) == e
    #     @test inv(A) * A == e
    #     @test B / A == B * inv(A)
    #     @test A \ B == inv(A) * B
    # end

    # Adjoints

    @test A'' == A

    # Operators act on functions

    @test A * (f + g) == A * f + A * g
    @test (A + B) * f == A * f + B * f

    for P3 in (Pr, Dl), R3 in 0:D
        F = rand(Op{D, P3, R3, P2, R2, T}, mf)
        @test (F * A) * f == F * (A * f)
    end
    
    # Add diagonal entries to help make F and G invertible
    F = one(Op{D, P1, R1, P1, R1, T}, mf) + rand(Op{D, P1, R1, P1, R1, T}, mf)
    G = one(Op{D, P1, R1, P1, R1, T}, mf) + rand(Op{D, P1, R1, P1, R1, T}, mf)

    # Note: \ converts rationals to Float64
    g1 = (G * F) \ f
    g2 = F \ (G \ f)
    maxabs(f) = norm(f.values, Inf)
    gscale = max(1, maxabs(g1), maxabs(g2))
    @test maxabs(g1 - g2) <= 1.0e-12 * gscale
end



@testset "Manifold ops D=$D P=$P R=$R" for D in 0:Dmax, P in (Pr, Dl), R in 0:D
    T = Rational{Int64}

    mf0 = DManifold(Val(D))
    mf1 = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    mf2 = hypercube_manifold(Val(D))
    mfs = [mf0, mf1, mf2]

    for mf in mfs
        f0 = ones(Fun{D, P, R, T}, mf)
        f1 = id(Fun{D, P, R, T}, mf)
        fs = [f0, f1]

        R1 = P == Pr ? R-1 : R+1
        if 0 <= R1 <= D
            b = boundary(Val(P), Val(R), mf)
            for f in fs
                bf = b*f
                bf::Fun{D, P, R1, T}
            end
        end

        R1 = P == Pr ? R+1 : R-1
        if 0 <= R1 <= D
            d = deriv(Val(P), Val(R), mf)
            for f in fs
                df = d*f
                df::Fun{D, P, R1, T}
            end
        end
    end
end
