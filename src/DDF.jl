module DDF

export Domain
struct Domain{D}
    nvertices::Int
    # vertices are always numbered 1:nvertices and are not stored
    edges::Vector{NTuple{2, Int}}
    faces::Vector{NTuple{3, Int}}

    function Domain{D}(nvertices::Int,
                       edges::Vector{NTuple{2, Int}},
                       faces::Vector{NTuple{3, Int}}) where {D}
        dom = new{D}(nvertices, edges, faces)
        @assert invariant(dom)
        dom
    end
end

export invariant
function invariant(dom::Domain{D})::Bool where {D}
    D >= 0 || return false

    dom.nvertices >= 0 || return false

    (D >= 1 || isempty(dom.edges)) || return false
    all(1 <= e[1] <= dom.nvertices for e in dom.edges) || return false
    all(1 <= e[2] <= dom.nvertices for e in dom.edges) || return false
    all(e[1] < e[2] for e in dom.edges) || return false
    all(dom.edges[i] < dom.edges[i+1]
        for i in 1:length(dom.edges)-1) || return false

    (D >= 2 || isempty(dom.faces)) || return false
    all(1 <= f[1] <= dom.nvertices for f in dom.faces) || return false
    all(1 <= f[2] <= dom.nvertices for f in dom.faces) || return false
    all(1 <= f[3] <= dom.nvertices for f in dom.faces) || return false
    all(f[1] < f[2] < f[3] for f in dom.faces) || return false
    all(dom.faces[i] < dom.faces[i+1]
        for i in 1:length(dom.faces)-1) || return false

    D <= 2 || return false

    return true
end

# Comparison

function Base.:(==)(dom1::Domain{D}, dom2::Domain{D})::Bool where {D}
    dom1.nvertices == dom2.nvertices || return false
    dom1.edges == dom2.edges || return false
    dom1.faces == dom2.faces || return false
    return true
end

export dim
function dim(::Val{0}, dom::Domain{D})::Int where {D}
    @assert 0 <= D
    dom.nvertices
end
function dim(::Val{1}, dom::Domain{D})::Int where {D}
    @assert 1 <= D
    length(dom.edges)
end
function dim(::Val{2}, dom::Domain{D})::Int where {D}
    @assert 2 <= D
    length(dom.faces)
end

export empty_domain
function empty_domain(::Val{D})::Domain{D} where {D}
    Domain{D}(0, NTuple{2, Int}[], NTuple{3, Int}[])
end

export cell_domain
function cell_domain(cell::NTuple{D1, Int})::Domain{D1-1} where {D1}
    D = D1 - 1
    @assert 0 <= D
    if D == 0
        nvertices = D + 1
        Domain{D}(nvertices, NTuple{2, Int}[], NTuple{3, Int}[])
    elseif D == 1
        nvertices = D + 1
        edges = [cell]
        Domain{D}(nvertices, edges, NTuple{3, Int}[])
    elseif D == 2
        nvertices = D + 1
        faces = [cell]
        edges = ([(cell[1], cell[2]), (cell[1], cell[3]), (cell[2], cell[3])])
        Domain{D}(nvertices, edges, faces)
    else
        @assert false
    end
end



export Fun
struct Fun{D, R, T}             # <: AbstractVector{T}
    dom::Domain{D}
    values::Vector{T}

    function Fun{D, R, T}(dom::Domain{D}, values::Vector{T}) where {D, R, T}
        fun = new{D, R, T}(dom, values)
        @assert invariant(fun)
        fun
    end
end

export invariant
function invariant(fun::Fun{D, R, T})::Bool where {D, R, T}
    0 <= R <= D || return false
    invariant(fun.dom) || return false
    length(fun.values) == dim(Val(R), fun.dom) || return false
    return true
end

# Comparison

function Base.:(==)(f::Fun{D, R, T}, g::Fun{D, R, T})::Bool where {D, R, T}
    @assert f.dom == g.dom
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
    @assert all(f.dom == g.dom for g in gs)
    rvalues = map(op, f.values, (g.values for g in gs)...)
    U = eltype(rvalues)
    Fun{D, R, U}(f.dom, rvalues)
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
                   dom::Domain{D})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(dom, zeros(T, dim(Val(R), dom)))
end
function Base.one(::Type{Fun{D, R, T}},
                  dom::Domain{D})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(dom, ones(T, dim(Val(R), dom)))
end

export id
function id(::Type{Fun{D, R, T}},
            dom::Domain{D})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(dom, T[T(i) for i in 1:dim(Val(R), dom)])
end

function Base.:+(f::Fun{D, R}) where {D, R}
    rvalues = +f.values
    U = eltype(rvalues)
    Fun{D, R, U}(f.dom, rvalues)
end

function Base.:-(f::Fun{D, R, T})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.dom, -f.values)
end

function Base.:+(f::Fun{D, R, T}, g::Fun{D, R, T})::Fun{D, R, T} where {D, R, T}
    @assert f.dom == g.dom
    Fun{D, R, T}(f.dom, f.values + g.values)
end

function Base.:-(f::Fun{D, R, T}, g::Fun{D, R, T})::Fun{D, R, T} where {D, R, T}
    @assert f.dom == g.dom
    Fun{D, R, T}(f.dom, f.values - g.values)
end

function Base.:*(a::T, f::Fun{D, R, T})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.dom, a * f.values)
end

function Base.:*(f::Fun{D, R, T}, a::T)::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.dom, f.values * a)
end

function Base.:\(a::T, f::Fun{D, R, T})::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.dom, a \ f.values)
end

function Base.:/(f::Fun{D, R, T}, a::T)::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.dom, f.values / a)
end

function Base.conj(f::Fun{D, R, T}, a::T)::Fun{D, R, T} where {D, R, T}
    Fun{D, R, T}(f.dom, conj(f.values))
end

end
