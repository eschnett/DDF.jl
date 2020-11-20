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

    if N ≥ 2
        xs = SVector{N}(xs[2], xs[1], xs[3:end]...)
        v = volume(xs)
        @test v == T(1) / factorial(N - 1)
    end

    if N ≥ 2
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

@testset "Morton ordering D=$D" for D in 1:Dmax
    # nbits = 10 ÷ D
    nbits = 2 ÷ D
    imax = (1 << nbits) - 1
    found = falses(1 << (D * nbits))
    for i in
        CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> imax, D))
        u = SVector{D,UInt64}(i.I)
        m = morton(u)
        @test !found[m + 1]
        found[m + 1] = true
    end

    T = Float64
    xmin = SVector{D,T}(-1 for d in 1:D)
    xmax = SVector{D,T}(+1 for d in 1:D)
    dx = (xmax - xmin) / (imax .+ 1)
    ifound = UInt64[]
    for i in
        CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> imax, D))
        u = SVector{D,UInt64}(i.I)
        # x = xmin + u .* dx + (T(0.01) .+ T(0.98) * rand(SVector{D,T}) .* dx)
        x = xmin + (u + rand(SVector{D,T})) .* dx
        @assert all(xmin .≤ x .≤ xmax)
        m = morton(x, xmin, xmax)
        push!(ifound, m)
    end
    @test allunique(sort!(ifound))
end
