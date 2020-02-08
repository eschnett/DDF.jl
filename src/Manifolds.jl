export Manifold
struct Manifold{D}
    nvertices::Int
    # vertices are always numbered 1:nvertices and are not stored
    edges::Vector{NTuple{2, Int}}
    faces::Vector{NTuple{3, Int}}

    function Manifold{D}(nvertices::Int,
                       edges::Vector{NTuple{2, Int}},
                       faces::Vector{NTuple{3, Int}}) where {D}
        mf = new{D}(nvertices, edges, faces)
        @assert invariant(mf)
        mf
    end
end

export invariant
function invariant(mf::Manifold{D})::Bool where {D}
    D >= 0 || return false

    mf.nvertices >= 0 || return false

    (D >= 1 || isempty(mf.edges)) || return false
    all(1 <= e[1] <= mf.nvertices for e in mf.edges) || return false
    all(1 <= e[2] <= mf.nvertices for e in mf.edges) || return false
    all(e[1] < e[2] for e in mf.edges) || return false
    all(mf.edges[i] < mf.edges[i+1]
        for i in 1:length(mf.edges)-1) || return false

    (D >= 2 || isempty(mf.faces)) || return false
    all(1 <= f[1] <= mf.nvertices for f in mf.faces) || return false
    all(1 <= f[2] <= mf.nvertices for f in mf.faces) || return false
    all(1 <= f[3] <= mf.nvertices for f in mf.faces) || return false
    all(f[1] < f[2] < f[3] for f in mf.faces) || return false
    all(mf.faces[i] < mf.faces[i+1]
        for i in 1:length(mf.faces)-1) || return false

    D <= 2 || return false

    return true
end

# Comparison

function Base.:(==)(mf1::Manifold{D}, mf2::Manifold{D})::Bool where {D}
    mf1.nvertices == mf2.nvertices || return false
    mf1.edges == mf2.edges || return false
    mf1.faces == mf2.faces || return false
    return true
end

export dim
function dim(::Val{0}, mf::Manifold{D})::Int where {D}
    @assert 0 <= D
    mf.nvertices
end
function dim(::Val{1}, mf::Manifold{D})::Int where {D}
    @assert 1 <= D
    length(mf.edges)
end
function dim(::Val{2}, mf::Manifold{D})::Int where {D}
    @assert 2 <= D
    length(mf.faces)
end

export empty_manifold
function empty_manifold(::Val{D})::Manifold{D} where {D}
    Manifold{D}(0, NTuple{2, Int}[], NTuple{3, Int}[])
end

export cell_manifold
function cell_manifold(cell::NTuple{D1, Int})::Manifold{D1-1} where {D1}
    D = D1 - 1
    @assert 0 <= D
    if D == 0
        nvertices = D + 1
        Manifold{D}(nvertices, NTuple{2, Int}[], NTuple{3, Int}[])
    elseif D == 1
        nvertices = D + 1
        edges = [cell]
        Manifold{D}(nvertices, edges, NTuple{3, Int}[])
    elseif D == 2
        nvertices = D + 1
        faces = [cell]
        edges = ([(cell[1], cell[2]), (cell[1], cell[3]), (cell[2], cell[3])])
        Manifold{D}(nvertices, edges, faces)
    else
        @assert false
    end
end
