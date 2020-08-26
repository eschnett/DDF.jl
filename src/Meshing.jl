module Meshing

using Delaunay
using SparseArrays
using StaticArrays

using ..SparseOps
using ..ZeroOrOne

export delaunay_mesh
function delaunay_mesh(coords::Array{T,2}) where {T}
    D = size(coords, 2)
    N = D + 1
    nvertices = size(coords, 1)

    if D == 0
        @assert nvertices == 1
        simplices = sparse([1], [1], [One()], 1, 1)
        return simplices
    end

    # Triangulate
    mesh = delaunay(coords)

    # Convert to sparse matrix
    nsimplices = size(mesh.simplices, 1)
    @assert size(mesh.simplices, 2) == N
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

end
