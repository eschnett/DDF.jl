using DDF

using DifferentialForms
using StaticArrays
using Test

@testset "Circumcentre D=$D N=$N" for D in 1:Dmax, N in 1:(D + 1)
    T = Rational{BigInt}
    xs = SVector{N}(rand(Form{D,1,T}) for n in 1:N)
    cc = circumcentre(xs)
    cc::Form{D,1,T}
    # Check that circumcentre has same distance from all vertices
    r2 = norm2(xs[1] - cc)
    @test all(==(r2), norm2(xs[n] - cc) for n in 1:N)
    # Check that circumcentre lies in hyperplane spanned by vertices
    if sum(frank.(xs)) > D
        @test true
    else
        subspace = ∧(xs...)
        if frank(cc) + frank(subspace) > D
            @test true
        else
            @test iszero(cc ∧ subspace)
        end
    end
end

@testset "Weighted circumcentre D=$D N=$N" for D in 1:Dmax, N in 1:(D + 1)
    T = Rational{BigInt}
    xs = SVector{N}(rand(Form{D,1,T}) for n in 1:N)
    ws = SVector{N}(rand(Form{D,0,T}) for n in 1:N)
    cc = circumcentre(xs, ws)
    cc::Form{D,1,T}
    # Check that circumcentre has correct weighted distance from all vertices
    r2 = norm2(xs[1] - cc) - ws[1][]
    @test all(==(r2), norm2(xs[n] - cc) - ws[n][] for n in 1:N)
    # Check that circumcentre lies in hyperplane spanned by vertices
    if sum(frank.(xs)) > D
        @test true
    else
        subspace = ∧(xs...)
        if frank(cc) + frank(subspace) > D
            @test true
        else
            @test iszero(cc ∧ subspace)
        end
    end
end

@testset "Volume D=$D N=$N" for D in 1:Dmax, N in 1:(D + 1)
    T = Rational{BigInt}

    x0 = zero(Form{D,1,T})
    xs = SVector{N}(x0, (x0 + unit(Form{D,1,T}, n) for n in 1:(N - 1))...)
    v = volume(xs)
    @test v == T(1) / factorial(N - 1)

    xs = SVector{N}(x0,
                    (x0 + unit(Form{D,1,T}, n - (N - 1) + D) for n in 1:(N - 1))...)
    v = volume(xs)
    @test v == T(1) / factorial(N - 1)

    if N >= 2
        xs = SVector{N}(xs[2], xs[1], xs[3:end]...)
        v = volume(xs)
        @test v == T(1) / factorial(N - 1)
    end

    if N >= 2
        xs = SVector{N}(xs[2:end]..., xs[1])
        v = volume(xs)
        @test v == T(1) / factorial(N - 1)
    end

    if N > 1
        xs = SVector{N}(x0,
                        (x0 + unit(Form{D,1,T}, n - (N - 1) + D) for n in
                                                                     1:(N - 1))...)
        xs = SVector{N}(xs[1:(end - 1)]..., 5 * xs[end])
        v = volume(xs)
        @test v == T(5) / factorial(N - 1)
    end

    x1 = rand(Form{D,1,T})
    xs = SVector{N}(x1,
                    (x1 + unit(Form{D,1,T}, n - (N - 1) + D) for n in 1:(N - 1))...)
    v = volume(xs)
    @test v == T(1) / factorial(N - 1)

    xs = 2 * xs
    v = volume(xs)
    @test v == T(2^(N - 1)) / factorial(N - 1)
end

#TODO @testset "Dual volume" begin end
