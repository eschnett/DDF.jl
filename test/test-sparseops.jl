using DDF

using LinearAlgebra
using SparseArrays
using StaticArrays
using Test

@testset "Sparse matrix iteration" begin
    nrows = 11
    ncols = 12
    A = sprand(Complex{Int8}, nrows, ncols, 0.25)
    nvals = 0
    for col in 1:ncols
        for (row, val) in sparse_column(A, col)
            @test val == A[row, col]
            nvals += 1
        end
    end
    @test nvals == nnz(A)

    for col in 1:ncols
        Ac = sparse_column(A, col)
        Acr = sparse_column_rows(A, col)
        Acv = sparse_column_values(A, col)
        colvals = [row_val for row_val in Ac]
        colvals::Vector{Tuple{Int,Complex{Int8}}}
        @test length(Ac) == length(colvals)
        @test isempty(Ac) == (length(Ac) == 0)
        if !isempty(Ac)
            @test first(Ac) == colvals[1]
            @test last(Ac) == colvals[end]
        end
        for val in colvals
            @test val ∈ Ac
            @test .-val ∉ Ac
        end
        @test collect(Ac) == colvals
        @test collect(Acr) == first.(colvals)
        @test collect(Acv) == last.(colvals)
    end
end

@testset "SparseOp vector space operations" begin
    T = Rational{Int64}
    R1 = 1
    R2 = 2

    nelts1 = rand(5:10)
    nelts2 = rand(5:10)

    z = zero(SparseOp{R1,R2,T}, nelts1, nelts2)
    A = rand(SparseOp{R1,R2,T}, nelts1, nelts2)
    B = rand(SparseOp{R1,R2,T}, nelts1, nelts2)
    C = rand(SparseOp{R1,R2,T}, nelts1, nelts2)
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

    # Adjoints

    @test A'' == A
    @test (a * A)' == A' * a'
    @test (A + B)' == A' + B'
end

@testset "SparseOp category operations" begin
    T = Rational{Int64}
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4

    nelts1 = rand(5:10)
    nelts2 = rand(5:10)
    nelts3 = rand(5:10)
    nelts4 = rand(5:10)

    e = one(SparseOp{R1,R1,T}, nelts1, nelts1)
    e2 = one(SparseOp{R2,R2,T}, nelts2, nelts2)
    z = zero(SparseOp{R1,R2,T}, nelts1, nelts2)
    z2 = zero(SparseOp{R2,R3,T}, nelts2, nelts3)
    z3 = zero(SparseOp{R1,R3,T}, nelts1, nelts3)
    A = rand(SparseOp{R1,R2,T}, nelts1, nelts2)
    B = rand(SparseOp{R2,R3,T}, nelts2, nelts3)
    C = rand(SparseOp{R3,R4,T}, nelts3, nelts4)
    a = rand(T)

    @test e * A == A
    @test A * e2 == A

    @test z * B == z3
    @test A * z2 == z3

    @test nnz((z * B).op) == 0
    @test nnz((A * z2).op) == 0

    @test (A * B) * C == A * (B * C)

    @test a * (A * B) == (a * A) * B
    @test (A * B) * a == A * (B * a)

    # Adjoints

    @test (A * B)' == B' * A'
end
