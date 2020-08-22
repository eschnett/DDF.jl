module Defs

using LinearAlgebra
using SparseArrays
using StaticArrays

export invariant
function invariant end

export sparse_column
"""
All nonzero row indices and values for a column in a sparse matrix
"""
function sparse_column(A::SparseMatrixCSC, col::Integer)
    return SparseMatrixCSCColumnIter(Val(:RowVal), A, col)
end

export sparse_column_rows
"""
All row indices of nonzero values for a column in a sparse matrix
"""
function sparse_column_rows(A::SparseMatrixCSC, col::Integer)
    return SparseMatrixCSCColumnIter(Val(:Row), A, col)
end

export sparse_column_values
"""
All nonzero values for a column in a sparse matrix
"""
function sparse_column_values(A::SparseMatrixCSC, col::Integer)
    return SparseMatrixCSCColumnIter(Val(:Val), A, col)
end

struct SparseMatrixCSCColumnIter{F,T,I}
    rowvals::Vector{I}
    nonzeros::Vector{T}
    nzrange::UnitRange{Int}
    function SparseMatrixCSCColumnIter(::F, A::SparseMatrixCSC{T,I},
                                       col::Integer) where {F,T,I}
        new{F,T,I}(rowvals(A), nonzeros(A), nzrange(A, col))
    end
end

function process(iter::SparseMatrixCSCColumnIter{Val{:RowVal}}, ind)
    iter.rowvals[ind], iter.nonzeros[ind]
end
process(iter::SparseMatrixCSCColumnIter{Val{:Row}}, ind) = iter.rowvals[ind]
process(iter::SparseMatrixCSCColumnIter{Val{:Val}}, ind) = iter.nonzeros[ind]

Base.first(iter::SparseMatrixCSCColumnIter) = process(iter, first(iter.nzrange))
Base.last(iter::SparseMatrixCSCColumnIter) = process(iter, last(iter.nzrange))
Base.length(iter::SparseMatrixCSCColumnIter) = length(iter.nzrange)

function Base.iterate(iter::SparseMatrixCSCColumnIter, state...)
    ind_next = iterate(iter.nzrange, state...)
    ind_next === nothing && return nothing
    ind, next = ind_next
    return process(iter, ind), next
end

export show_sparse
"""
Show a sparse matrix with a convenient layout
"""
function show_sparse(io::IO, A::SparseMatrixCSC{T,I}) where {T,I}
    # Convert to CSR -- this is expensive!
    A = permutedims(A)
    println(io, "$(A.n)×$(A.m) SparseMatrixCSC{$I,$T} ",
            "with $(length(A.nzval)) stored entries:")
    return for i in 1:(A.n)
        jmin = A.colptr[i]
        jmax = A.colptr[i + 1] - 1
        if jmin <= jmax
            print(io, "  [$i]:")
            for j in jmin:jmax
                print(io, " [$(A.rowval[j])]=$(A.nzval[j])")
            end
            println(io)
        end
    end
end
show_sparse(A::SparseMatrixCSC) = show_sparse(stdout, A)

end
