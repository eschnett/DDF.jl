module SparseOps

using LinearAlgebra
using SparseArrays

using ..Defs

################################################################################

# Sparse matrix iterators

export sparse_column
"""
All nonzero row indices and values for a column in a sparse matrix
"""
function sparse_column(A::SparseMatrixCSC, col::Integer)
    return SparseMatrixCSCColumn(Val(:RowVal), A, col)
end

export sparse_column_rows
"""
All row indices of nonzero values for a column in a sparse matrix
"""
function sparse_column_rows(A::SparseMatrixCSC, col::Integer)
    return SparseMatrixCSCColumn(Val(:Row), A, col)
end

export sparse_column_values
"""
All nonzero values for a column in a sparse matrix
"""
function sparse_column_values(A::SparseMatrixCSC, col::Integer)
    return SparseMatrixCSCColumn(Val(:Val), A, col)
end

struct SparseMatrixCSCColumn{F,T,I}
    rowvals::Vector{I}
    nonzeros::Vector{T}
    nzrange::UnitRange{Int}
    function SparseMatrixCSCColumn(::F, A::SparseMatrixCSC{T,I},
                                   col::Integer) where {F,T,I}
        return new{F,T,I}(rowvals(A), nonzeros(A), nzrange(A, col))
    end
end

function process(iter::SparseMatrixCSCColumn{Val{:RowVal}}, ind)
    return iter.rowvals[ind], iter.nonzeros[ind]
end
process(iter::SparseMatrixCSCColumn{Val{:Row}}, ind) = iter.rowvals[ind]
process(iter::SparseMatrixCSCColumn{Val{:Val}}, ind) = iter.nonzeros[ind]

Base.eltype(::SparseMatrixCSCColumn{Val{:RowVal},T,I}) where {T,I} = Tuple{I,T}
Base.eltype(::SparseMatrixCSCColumn{Val{:Row},T,I}) where {T,I} = I
Base.eltype(::SparseMatrixCSCColumn{Val{:Val},T}) where {T} = T
Base.first(iter::SparseMatrixCSCColumn) = process(iter, first(iter.nzrange))
Base.getindex(iter::SparseMatrixCSCColumn, i) = process(iter, iter.nzrange[i])
Base.keys(iter::SparseMatrixCSCColumn) = iter.nzrange
Base.last(iter::SparseMatrixCSCColumn) = process(iter, last(iter.nzrange))
Base.length(iter::SparseMatrixCSCColumn) = length(iter.nzrange)

function Base.iterate(iter::SparseMatrixCSCColumn, state...)
    ind_next = iterate(iter.nzrange, state...)
    ind_next === nothing && return nothing
    ind, next = ind_next
    return process(iter, ind), next
end

################################################################################

export SparseOp
"""
A type-tagged sparse operator
"""
struct SparseOp{Tag1,Tag2,T}
    op::SparseMatrixCSC{T,Int}

    function SparseOp{Tag1,Tag2,T}(op::SparseMatrixCSC{T,Int}) where {Tag1,Tag2,
                                                                      T}
        return new{Tag1,Tag2,T}(op)
    end
    function SparseOp{Tag1,Tag2}(op::SparseMatrixCSC{T,Int}) where {Tag1,Tag2,T}
        return SparseOp{Tag1,Tag2,T}(op)
    end
end

################################################################################

# I/O

function Base.show(io::IO, A::SparseOp{Tag1,Tag2,T}) where {Tag1,Tag2,T}
    # Convert to CSR -- this is expensive!
    println(io, "$(size(A.op,1))×$(size(A.op,2)) SparseOp{$Tag1,$Tag2,$T} ",
            "with $(nnz(A.op)) stored entries:")
    println(io, "  col: rows")
    # Aop′ = permutedims(A.op)
    Aop′ = A.op
    for j in 1:size(Aop′, 2)
        print(io, "  [$j]: ")
        if !T.mutable && sizeof(T) == 0
            # Immutable types with size 0 have always the same value,
            # hence we don't need to output it
            print(io, "[")
            didoutput = false
            for i in sparse_column_rows(Aop′, j)
                didoutput && print(io, ", ")
                didoutput = true
                print(io, "$i")
            end
            print(io, "]")
        else
            didoutput = false
            for (i, v) in sparse_column(Aop′, j)
                didoutput && print(io, ", ")
                didoutput = true
                print(io, "[$i=$v]")
            end
        end
        println(io)
    end
end

################################################################################

# Comparisons

function Base.:(==)(A::SparseOp{Tag1,Tag2},
                    B::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    A === B && return true
    return A.op == B.op
end
function Base.:(<)(A::SparseOp{Tag1,Tag2},
                   B::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    return A.op < B.op
end
function Base.isequal(A::SparseOp{Tag1,Tag2},
                      B::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    return isequal(A.op, B.op)
end
function Base.hash(A::SparseOp{Tag1,Tag2}, h::UInt) where {Tag1,Tag2}
    return hash(0xf9ecec86, hash(Tag2, hash(Tag1, hash(A.op, h))))
end

function Base.rand(::Type{SparseOp{Tag1,Tag2,T}}, nrows::Integer,
                   ncols::Integer) where {Tag1,Tag2,T}
    p = clamp(4 / min(nrows, ncols), 0, 1)
    return SparseOp{Tag1,Tag2}(sprand(T, nrows, ncols, p))
end

################################################################################

# Collection

Base.eltype(::Type{<:SparseOp{<:Any,<:Any,T}}) where {T} = T
Base.eltype(A::SparseOp) = eltype(typeof(A))
Base.isempty(A::SparseOp) = isempty(A.op)
Base.iterate(A::SparseOp, state...) = iterate(A.op, state...)
Base.length(A::SparseOp) = length(A.op)
# function Base.map(f::F, A::SparseOp{Tag1,Tag2},
#                   Bs::SparseOp{Tag1,Tag2}...) where {F,Tag1,Tag2}
#     if nnz(A.op) == 0
#         U = typeof(f(one(eltype(A))))
#         return zero(SparseOp{Tag1,Tag2,U}, size(A, 1), size(A, 2))
#     end
#     SparseOp{Tag1,Tag2}(map(f, A.op, map(B -> B.op, Bs)...))
# end
function Base.map(f, A::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    U = typeof(f(one(eltype(A))))
    Bop = SparseMatrixCSC{U,Int}(A.op.m, A.op.n, A.op.colptr, A.op.rowval,
                                 map(f, A.op.nzval))
    return SparseOp{Tag1,Tag2}(Bop)
end
function Base.reduce(f, A::SparseOp{Tag1,Tag2}, Bs::SparseOp{Tag1,Tag2}...;
                     kw...) where {Tag1,Tag2}
    return reduce(f, A.op, map(B -> B.op, Bs)...; kw...)
end

sparse_column(A::SparseOp, col::Integer) = sparse_column(A.op, col)
sparse_column_rows(A::SparseOp, col::Integer) = sparse_column_rows(A.op, col)
function sparse_column_values(A::SparseOp, col::Integer)
    return sparse_column_values(A.op, col)
end

################################################################################

# Operators are an abstract matrix

Base.IndexStyle(::Type{<:SparseOp}) = IndexStyle(Vector)
Base.axes(A::SparseOp) = axes(A.op)
Base.axes(A::SparseOp, dir) = axes(A.op, dir)
Base.eachindex(A::SparseOp) = eachindex(A.op)
Base.getindex(A::SparseOp, inds...) = getindex(A.op, inds...)
Base.ndims(::SparseOp) = 2
Base.size(A::SparseOp) = size(A.op)
Base.size(A::SparseOp, dims) = size(A.op, dims)
Base.stride(A::SparseOp, k) = stride(A.op, k)
Base.strides(A::SparseOp) = strides(A.op)

################################################################################

# Vector space

function Base.zero(::Type{SparseOp{Tag1,Tag2,T}}, nrows::Integer,
                   ncols::Integer) where {Tag1,Tag2,T}
    return SparseOp{Tag1,Tag2}(spzeros(T, nrows, ncols))
end
Base.iszero(A::SparseOp) = iszero(A.op)

function Base.:+(A::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    return SparseOp{Tag1,Tag2}(+A.op)
end
function Base.:-(A::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    return SparseOp{Tag1,Tag2}(-A.op)
end

function Base.:+(A::SparseOp{Tag1,Tag2},
                 B::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    return SparseOp{Tag1,Tag2}(A.op + B.op)
end
function Base.:-(A::SparseOp{Tag1,Tag2},
                 B::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    return SparseOp{Tag1,Tag2}(A.op - B.op)
end

function Base.:*(a::Number, A::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    return SparseOp{Tag1,Tag2}(a * A.op)
end
function Base.:\(a::Number, A::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    return SparseOp{Tag1,Tag2}(a \ A.op)
end
function Base.:*(A::SparseOp{Tag1,Tag2}, a::Number) where {Tag1,Tag2}
    return SparseOp{Tag1,Tag2}(A.op * a)
end
function Base.:/(A::SparseOp{Tag1,Tag2}, a::Number) where {Tag1,Tag2}
    return SparseOp{Tag1,Tag2}(A.op / a)
end

################################################################################

# Algebra

function Base.one(::Type{SparseOp{Tag,Tag,T}}, nrows::Integer,
                  ncols::Integer) where {Tag,T}
    @assert nrows == ncols
    return SparseOp{Tag,Tag}(sparse(one(T) * I, nrows, ncols))
end
Base.isone(A::SparseOp{Tag,Tag}) where {Tag} = A.op == I

function Base.:*(A::SparseOp{Tag1,Tag2},
                 B::SparseOp{Tag2,Tag3}) where {Tag1,Tag2,Tag3}
    Cop = A.op * B.op
    if eltype(Cop) <: Integer
        dropzeros!(Cop)
    end
    return SparseOp{Tag1,Tag3}(Cop)
end

function Base.adjoint(A::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    return SparseOp{Tag2,Tag1}(permutedims(A.op))
end

end
