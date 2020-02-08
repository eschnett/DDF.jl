export Fun
struct Fun{D, R, T}             # <: AbstractVector{T}
    mf::Manifold{D}
    values::Vector{T}

    function Fun{D, R, T}(mf::Manifold{D}, values::Vector{T}) where {D, R, T}
        fun = new{D, R, T}(mf, values)
        @assert invariant(fun)
        fun
    end
end

export invariant
function invariant(fun::Fun{D, R, T})::Bool where {D, R, T}
    0 <= R <= D || return false
    invariant(fun.mf) || return false
    length(fun.values) == dim(Val(R), fun.mf) || return false
    return true
end

# Comparison

function Base.:(==)(f::Fun{D, R, T}, g::Fun{D, R, T})::Bool where {D, R, T}
    @assert f.mf == g.mf
    f.values == g.values
end

# Functions are a collection

Base.iterate(f::Fun, state...) = iterate(f.values, state...)
Base.IteratorSize(::Fun) = Base.IteratorSize(Vector)
Base.IteratorEltype(::Fun) = Base.IteratorEltype(Vector)
Base.isempty(f::Fun) = isempty(f.values)
Base.length(f::Fun) = length(f.values)
Base.eltype(::Type{Fun{D, R, T}}) where {D, R, T} = eltype(Vector{T})

function Base.map(op, f::Fun{D, R}, gs::Fun{D, R}...) where {D, R}
    @assert all(f.mf == g.mf for g in gs)
    rvalues = map(op, f.values, (g.values for g in gs)...)
    U = eltype(rvalues)
    Fun{D, R, U}(f.mf, rvalues)
end

# Functions are an abstract vector

Base.ndims(::Fun) = 1
Base.size(f::Fun) = size(f.values)
Base.size(f::Fun, dims) = size(f.values, dims)
Base.axes(f::Fun) = axes(f.values)
Base.axes(f::Fun, dir) = axes(f.values, dir)
Base.eachindex(f::Fun) = eachindex(f.values)
Base.IndexStyle(::Type{<:Fun}) = IndexStyle(Vector)
Base.stride(f::Fun, k) = stride(f.values, k)
Base.strides(f::Fun) = strides(f.values)
Base.getindex(f::Fun, inds...) = getindex(f.values, inds...)

# Functions are a vector space

function Base.zero(::Type{Fun{D, R, T}},
                   mf::Manifold{D})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(mf, zeros(T, dim(Val(R), mf)))
end
function Base.one(::Type{Fun{D, R, T}},
                  mf::Manifold{D})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(mf, ones(T, dim(Val(R), mf)))
end

export id
function id(::Type{Fun{D, R, T}},
            mf::Manifold{D})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(mf, T[T(i) for i in 1:dim(Val(R), mf)])
end

function Base.:+(f::Fun{D, R}) where {D, R}
    rvalues = +f.values
    U = eltype(rvalues)
    Fun{D, R, U}(f.mf, rvalues)
end

function Base.:-(f::Fun{D, R, T})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.mf, -f.values)
end

function Base.:+(f::Fun{D, R, T}, g::Fun{D, R, T})::Fun{D, R, T} where {D, R, T}
    @assert f.mf == g.mf
    Fun{D, R, T}(f.mf, f.values + g.values)
end

function Base.:-(f::Fun{D, R, T}, g::Fun{D, R, T})::Fun{D, R, T} where {D, R, T}
    @assert f.mf == g.mf
    Fun{D, R, T}(f.mf, f.values - g.values)
end

function Base.:*(a::T, f::Fun{D, R, T})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.mf, a * f.values)
end

function Base.:*(f::Fun{D, R, T}, a::T)::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.mf, f.values * a)
end

function Base.:\(a::T, f::Fun{D, R, T})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.mf, a \ f.values)
end

function Base.:/(f::Fun{D, R, T}, a::T)::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.mf, f.values / a)
end

function Base.conj(f::Fun{D, R, T}, a::T)::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.mf, conj(f.values))
end
