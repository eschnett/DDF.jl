module SparseOps

using LinearAlgebra
using SparseArrays
using StaticArrays

using ..Defs

################################################################################

export ID
"""
Tagged ID (or index) for various IDVector and SparseOp

This type is a pure ordinal; it cannot be used for counting.
"""
struct ID{Tag}
    id::Int
end

Base.Int(i::ID) = i.id
Base.convert(::Type{Int}, i::ID) = Int(i)

const subscripts = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
Base.show(io::IO, i::ID{Tag}) where {Tag} = print(io, i.id, subscripts[Tag + 1])

Base.:(==)(i::ID{Tag}, j::ID{Tag}) where {Tag} = i.id == j.id
Base.:(==)(::ID, ::ID) = false
Base.:(<)(i::ID{Tag}, j::ID{Tag}) where {Tag} = i.id < j.id
Base.isless(i::ID{Tag}, j::ID{Tag}) where {Tag} = isless(i.id, j.id)
function Base.hash(i::ID{Tag}, h::UInt) where {Tag}
    return hash(0x4e3ca100, hash(Tag, hash(i.id, h)))
end

Base.zero(::Type{ID{Tag}}) where {Tag} = ID{Tag}(0)
Base.:+(i::ID{Tag}, j::Int) where {Tag} = ID{Tag}(i.id + j)
Base.:-(i::ID{Tag}, j::Int) where {Tag} = ID{Tag}(i.id - j)
Base.:-(i::ID{Tag}, j::ID{Tag}) where {Tag} = i.id - j.id

################################################################################

export IDVector
struct IDVector{Tag,T}
    vec::Vector{T}
end

IDVector{Tag}(xs::Vector{T}) where {Tag,T} = IDVector{Tag,T}(xs)
IDVector{Tag,T}() where {Tag,T} = IDVector{Tag}(T[])

function Base.show(io::IO, xs::IDVector{Tag}) where {Tag}
    return print(io, xs.vec, subscripts[Tag + 1])
end

Base.:(==)(xs::IDVector{Tag}, ys::IDVector{Tag}) where {Tag} = xs.vec == ys.vec
Base.:(==)(::IDVector, ::IDVector) = false
function Base.isless(xs::IDVector{Tag}, y::IDVector{Tag}) where {Tag}
    return isless(xs.vec, ys.vec)
end
function Base.hash(xs::IDVector{Tag}, h::UInt) where {Tag}
    return hash(0x934f6884, hash(Tag, hash(xs.vec, h)))
end

Base.axes(xs::IDVector{Tag}, d) where {Tag} = ID{Tag}(1):ID{Tag}(length(xs))
Base.axes(xs::IDVector) = (axes(xs, 1),)
Base.eltype(::Type{<:IDVector{<:Any,T}}) where {T} = T
Base.first(xs::IDVector) = first(xs.vec)
Base.@propagate_inbounds function Base.getindex(xs::IDVector{Tag},
                                                i::ID{Tag}) where {Tag}
    return getindex(xs.vec, i.id)
end
Base.@propagate_inbounds function Base.getindex(xs::IDVector{Tag2},
                                                is::IDVector{Tag1,ID{Tag2}}) where {Tag1,
                                                                                    Tag2}
    return IDVector{Tag1}([xs[i] for i in is])
end
Base.@propagate_inbounds function Base.getindex(xs::IDVector{Tag,T},
                                                is::SVector{N,ID{Tag}}) where {T,
                                                                               N,
                                                                               Tag}
    return SVector{N,T}(Tuple(xs[i] for i in is))
end
Base.keys(xs::IDVector) = axes(xs, 1)
Base.last(xs::IDVector) = last(xs.vec)
Base.length(xs::IDVector) = length(xs.vec)
Base.ndims(::IDVector) = 1
Base.ndims(::Type{<:IDVector}) = 1
Base.@propagate_inbounds function Base.setindex!(xs::IDVector{Tag}, val,
                                                 i::ID{Tag}) where {Tag}
    return setindex!(xs.vec, val, i.id)
end
Base.size(xs::IDVector) = (length(xs),)
Base.size(xs::IDVector, d) = d == 1 ? length(xs) : 1

Base.pop!(xs::IDVector) = pop!(xs.vec)
Base.popfirst!(xs::IDVector) = popfirst!(xs.vec)
Base.push!(xs::IDVector, vals...) = (push!(xs.vec, vals...); xs)
Base.pushfirst!(xs::IDVector, vals...) = (pushfirst!(xs.vec, vals...); xs)

Base.copy(xs::IDVector{Tag,T}) where {Tag,T} = IDVector{Tag,T}(copy(xs.vec))
Base.iterate(xs::IDVector, state...) = iterate(xs.vec, state...)
Base.map(f, xs::IDVector{Tag}) where {Tag} = IDVector{Tag}(map(f, xs.vec))
function Base.map(f, xs::IDVector{Tag}, yss::IDVector{Tag}...) where {Tag}
    return IDVector{Tag}(map(f, xs.vec, (ys.vec for ys in yss)...))
end
Base.mapreduce(f, op, xs::IDVector; kws...) = mapreduce(f, op, xs.vec; kws...)
Base.reduce(f, xs::IDVector; kws...) = reduce(f, xs.vec; kws...)

Base.:+(xs::IDVector{Tag}) where {Tag} = IDVector{Tag}(+xs.vec)
Base.:-(xs::IDVector{Tag}) where {Tag} = IDVector{Tag}(-xs.vec)
function Base.:+(xs::IDVector{Tag}, ys::IDVector{Tag}) where {Tag}
    return IDVector{Tag}(xs.vec + ys.vec)
end
function Base.:-(xs::IDVector{Tag}, ys::IDVector{Tag}) where {Tag}
    return IDVector{Tag}(xs.vec - ys.vec)
end
Base.:*(a::Number, xs::IDVector{Tag}) where {Tag} = IDVector{Tag}(a * xs.vec)
Base.:\(a::Number, xs::IDVector{Tag}) where {Tag} = IDVector{Tag}(a \ xs.vec)
Base.:*(xs::IDVector{Tag}, a::Number) where {Tag} = IDVector{Tag}(xs.vec * a)
Base.:/(xs::IDVector{Tag}, a::Number) where {Tag} = IDVector{Tag}(xs.vec / a)

################################################################################

# Sparse matrix constructors

export MakeSparse
struct MakeSparse{Tv,Iv}
    m::Iv
    n::Iv
    combine::Any
    I::Vector{Iv}
    J::Vector{Iv}
    V::Vector{Tv}
    function MakeSparse{Tv,Iv}(m::Integer, n::Integer,
                               combine=nothing) where {Tv,Iv}
        return new{Tv,Iv}(m, n, combine, Vector{Iv}(), Vector{Iv}(),
                          Vector{Tv}())
    end
    function MakeSparse{Tv}(m::Integer, n::Integer, combine=nothing) where {Tv}
        return MakeSparse{Tv,Int}(m, n, combine)
    end
end

function SparseArrays.sparse(ms::MakeSparse)
    combines = ms.combine ≡ nothing ? () : (combine,)
    return sparse(ms.I, ms.J, ms.V, ms.m, ms.n, combines...)
end

Base.size(ms::MakeSparse) = (ms.m, ms.n)

Base.@propagate_inbounds function Base.setindex!(ms::MakeSparse, v, i::Integer,
                                                 j::Integer)
    @assert 1 ≤ i ≤ ms.m
    @assert 1 ≤ j ≤ ms.n
    push!(ms.I, i)
    push!(ms.J, j)
    push!(ms.V, v)
    return v
end

################################################################################

# Sparse matrix iterators

export sparse_column
"""
All nonzero row indices and values for a column ∈ a sparse matrix
"""
function sparse_column(A::SparseMatrixCSC{T,I}, col::Integer) where {T,I}
    return SparseMatrixCSCColumn(Val(:RowVal), I, A, col)
end

# TODO: write this as `keys(sparse_column(...))`
export sparse_column_rows
"""
All row indices of nonzero values for a column ∈ a sparse matrix
"""
function sparse_column_rows(A::SparseMatrixCSC{T,I}, col::Integer) where {T,I}
    return SparseMatrixCSCColumn(Val(:Row), I, A, col)
end

export sparse_column_values
"""
All nonzero values for a column ∈ a sparse matrix
"""
function sparse_column_values(A::SparseMatrixCSC{T,I}, col::Integer) where {T,I}
    return SparseMatrixCSCColumn(Val(:Val), I, A, col)
end

struct SparseMatrixCSCColumn{F,Idx,T,I} <: AbstractVector{T}
    rowvals::Vector{I}
    nonzeros::Vector{T}
    nzrange::UnitRange{Int}
    function SparseMatrixCSCColumn(::F, ::Type{Idx}, A::SparseMatrixCSC{T,I},
                                   col::Integer) where {F,Idx,T,I}
        return new{F,Idx,T,I}(rowvals(A), nonzeros(A), nzrange(A, col))
    end
end

function process(iter::SparseMatrixCSCColumn{Val{:RowVal},Idx}, ind) where {Idx}
    return Idx(iter.rowvals[ind]), iter.nonzeros[ind]
end
function process(iter::SparseMatrixCSCColumn{Val{:Row},Idx}, ind) where {Idx}
    return Idx(iter.rowvals[ind])
end
process(iter::SparseMatrixCSCColumn{Val{:Val}}, ind) = iter.nonzeros[ind]

function Base.eltype(::SparseMatrixCSCColumn{Val{:RowVal},Idx,T,I}) where {Idx,
                                                                           T,I}
    return Tuple{Idx,T}
end
Base.eltype(::SparseMatrixCSCColumn{Val{:Row},Idx,T,I}) where {Idx,T,I} = Idx
Base.eltype(::SparseMatrixCSCColumn{Val{:Val},Idx,T}) where {Idx,T} = T
Base.first(iter::SparseMatrixCSCColumn) = process(iter, first(iter.nzrange))
Base.@propagate_inbounds function Base.getindex(iter::SparseMatrixCSCColumn, i)
    return process(iter, iter.nzrange[i])
end
Base.keys(iter::SparseMatrixCSCColumn) = iter.nzrange
Base.last(iter::SparseMatrixCSCColumn) = process(iter, last(iter.nzrange))
Base.length(iter::SparseMatrixCSCColumn) = length(iter.nzrange)
Base.size(iter::SparseMatrixCSCColumn, i) = length(iter)
Base.size(iter::SparseMatrixCSCColumn) = (length(iter),)

function Base.iterate(iter::SparseMatrixCSCColumn, state...)
    ind_next = iterate(iter.nzrange, state...)
    ind_next ≡ nothing && return nothing
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
end

function SparseOp{Tag1,Tag2}(op::SparseMatrixCSC{T,Int}) where {Tag1,Tag2,T}
    return SparseOp{Tag1,Tag2,T}(op)
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
                print(io, "[$i:$v]")
            end
        end
        println(io)
    end
end

################################################################################

# Comparisons

function Base.:(==)(A::SparseOp{Tag1,Tag2},
                    B::SparseOp{Tag1,Tag2}) where {Tag1,Tag2}
    A ≡ B && return true
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

function sparse_column(A::SparseOp{Tag1,Tag2}, col::ID{Tag2}) where {Tag1,Tag2}
    return SparseMatrixCSCColumn(Val(:RowVal), ID{Tag1}, A.op, col.id)
end
function sparse_column_rows(A::SparseOp{Tag1,Tag2},
                            col::ID{Tag2}) where {Tag1,Tag2}
    return SparseMatrixCSCColumn(Val(:Row), ID{Tag1}, A.op, col.id)
end
function sparse_column_values(A::SparseOp{Tag1,Tag2},
                              col::ID{Tag2}) where {Tag1,Tag2}
    return SparseMatrixCSCColumn(Val(:Val), ID{Tag1}, A.op, col.id)
end

################################################################################

# Operators are an abstract matrix

Base.IndexStyle(::Type{<:SparseOp}) = IndexCartsian()
Base.axes(A::SparseOp) = ntuple(dir -> axes(A.op, dir), ndims(A))
@inline function Base.axes(A::SparseOp{Tag1,Tag2}, dir) where {Tag1,Tag2}
    dir == 1 && return ID{Tag1}(1):ID{Tag1}(size(A, 1))
    dir == 2 && return ID{Tag2}(1):ID{Tag2}(size(A, 2))
    return throw(DimensionMismatch())
end
Base.eachindex(A::SparseOp) = CartesianIndices(axes(A))
Base.@propagate_inbounds function Base.getindex(A::SparseOp, inds...)
    return throw(ArgumentError())
end
Base.@propagate_inbounds function Base.getindex(A::SparseOp{Tag1,Tag2},
                                                i::ID{Tag1},
                                                j::ID{Tag2}) where {Tag1,Tag2}
    return getindex(A.op, i.id, j.id)
end
Base.ndims(::SparseOp) = 2
Base.size(A::SparseOp) = size(A.op)
Base.size(A::SparseOp, dir) = size(A.op, dir)

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

function Base.:*(A::SparseOp{Tag1,Tag2}, x::IDVector{Tag2}) where {Tag1,Tag2}
    return IDVector{Tag1}(A.op * x.vec)
end
function Base.:\(A::SparseOp{Tag1,Tag2}, x::IDVector{Tag1}) where {Tag1,Tag2}
    return IDVector{Tag2}(A.op \ x.vec)
end

end
