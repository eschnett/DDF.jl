module Meshing

using Delaunay
using SparseArrays
using StaticArrays

using ..SparseOps
using ..ZeroOrOne

################################################################################

export delaunay_mesh
"""
Find the Delaunay triangulation for a set of points
"""
function delaunay_mesh(coords::Vector{SVector{C,S}}) where {C,S}
    nvertices = length(coords)

    if C == 0
        @assert nvertices == 1
        simplices = sparse([1], [1], [One()], 1, 1)
        return simplices
    end

    # Triangulate
    mesh = delaunay(S[coords[i][c] for i in 1:nvertices, c in 1:C])

    # Convert to sparse matrix
    nsimplices = size(mesh.simplices, 1)
    @assert size(mesh.simplices, 2) == C + 1
    I = Int[]
    J = Int[]
    V = One[]
    for j in 1:nsimplices
        for i in @view mesh.simplices[j, :]
            push!(I, i)
            push!(J, j)
            push!(V, One())
        end
    end
    simplices = sparse(I, J, V, nvertices, nsimplices)

    return simplices
end

################################################################################

export refine_coords
"""
Refine a mesh
"""
function refine_coords(oldedges::SparseOp{0,1,One},
                       oldcoords::Vector{SVector{C,S}}) where {C,S}
    noldvertices, noldedges = size(oldedges)
    @assert length(oldcoords) == noldvertices
    nvertices = noldvertices + noldedges
    coords = copy(oldcoords)
    # Loop over all old edges
    for i in 1:noldedges
        si = sparse_column_rows(oldedges, i)
        @assert length(si) == 2
        x = sum(coords[j] for j in si) / length(si)
        push!(coords, x)
    end
    return coords
end

end
