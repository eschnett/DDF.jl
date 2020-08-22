module Defs

using LinearAlgebra
using SparseArrays
using StaticArrays

export invariant
function invariant end

export show_sparse
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
