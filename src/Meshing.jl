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
    # [:Qbb, :Qc, :Qz, :Q12, :QJ]

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
    # @assert all(i -> 1 ≤ i ≤ nvertices, I)
    # @assert all(j -> 1 ≤ j ≤ nsimplices, J)
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
        # x1 = coords[si[1]]
        # x2 = coords[si[2]]
        # # x = (x0+x1)/2
        # q = S(0.375) + S(0.25) * rand(S)
        # x = q * x1 + (q - 1) * x2
        push!(coords, x)
    end
    return coords
end

end
