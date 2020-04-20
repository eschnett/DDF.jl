export Manifold
"""
Manifold (aka Chain)
"""
struct Manifold{D}
    nvertices::Int
    # vertices are always numbered 1:nvertices and are not stored (or
    # should they?)
    # TODO: Store these as matrices instead? Or a tuples of vectors?
    # As sparse matrices?
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
# TODO: Implement also the "cube complex" representation

export invariant
function invariant(mf::Manifold{D})::Bool where {D}
    D >= 0 || return false

    mf.nvertices >= 0 || (@assert false; return false)

    (D >= 1 || isempty(mf.edges)) || (@assert false; return false)
    all(1 <= e[1] <= mf.nvertices for e in mf.edges) || (@assert false; return false)
    all(1 <= e[2] <= mf.nvertices for e in mf.edges) || (@assert false; return false)
    all(e[1] < e[2] for e in mf.edges) || (@assert false; return false)
    all(mf.edges[i] < mf.edges[i+1]
        for i in 1:length(mf.edges)-1) || (@assert false; return false)

    (D >= 2 || isempty(mf.faces)) || (@assert false; return false)
    all(1 <= f[1] <= mf.nvertices for f in mf.faces) || (@assert false; return false)
    all(1 <= f[2] <= mf.nvertices for f in mf.faces) || (@assert false; return false)
    all(1 <= f[3] <= mf.nvertices for f in mf.faces) || (@assert false; return false)
    all(f[1] < f[2] < f[3] for f in mf.faces) || (@assert false; return false)
    all(mf.faces[i] < mf.faces[i+1]
        for i in 1:length(mf.faces)-1) || (@assert false; return false)

    D <= 2 || (@assert false; return false)

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

# Simple constructors

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

function tuplesort(xs::NTuple{D, T})::NTuple{D, T} where {D, T}
    # TODO: Make this efficient
    tuple(sort(collect(xs))...)
end

export simplicial_manifold
function simplicial_manifold(cells::Vector{NTuple{D1, Int}}
                             )::Manifold{D1-1} where {D1}
    D = D1 - 1
    @assert 0 <= D
    # # Ensure cells are sorted
    # for i in 2:length(cells)
    #     @assert cells[i] > cells[i-1]
    # end
    # # Ensure cell vertices are sorted
    # for c in cells
    #     for d in 2:D+1
    #         @assert c[d] > c[d-1]
    #     end
    # end
    # Count vertices
    nvertices = 0
    for c in cells
        for d in 1:D+1
            nvertices = max(nvertices, c[d])
        end
    end
    # Ensure all vertices are mentioned (we could omit this check)
    vertices = falses(nvertices)
    for c in cells
        for d in 1:D+1
            vertices[c[d]] = true
        end
    end
    @assert all(vertices)
    # Determine edges
    if D < 1
        edges = NTuple{2, Int}[]
    elseif D == 1
        edges = cells
    elseif D > 1
        edges = NTuple{2, Int}[]
        for c in cells
            for d1 in 1:D+1, d2 in d1+1:D+1
                push!(edges, (c[d1], c[d2]))
            end
        end
    end
    for i in 1:length(edges)
        edges[i] = tuplesort(edges[i])
    end
    sort!(edges)
    unique!(edges)
    # Determine faces
    if D < 2
        faces = NTuple{3, Int}[]
    elseif D == 2
        faces = cells
    elseif D > 2
        faces = NTuple{3, Int}[]
        for c in cells
            for d1 in 1:D+1, d2 in d1+1:D+1, d3 in d2+1:D+1
                push!(faces, (c[d1], c[d2], c[d3]))
            end
        end
    end
    for i in 1:length(faces)
        faces[i] = tuplesort(faces[i])
    end
    sort!(faces)
    unique!(faces)
    # if D < 3
    #     cells = NTuple{4, Int}[]
    # elseif D == 3
    #     sort!(cells)
    #     unique!(cells)
    # elseif D > 3
    #     @assert false
    # end
    # Create manifold
    Manifold{D}(nvertices, edges, faces)
end

# Boundaries and derivatives

