using DDF

using SparseArrays

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
