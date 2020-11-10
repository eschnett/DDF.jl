module Meshing

using Delaunay
using DifferentialForms
# using MiniQhull
using SparseArrays
using StaticArrays

using ..Algorithms
using ..SparseOps
using ..ZeroOrOne

################################################################################

export delaunay_mesh
"""
Find the Delaunay triangulation for a set of points
"""
function delaunay_mesh(coords::IDVector{0,SVector{C,S}}) where {C,S}
    nvertices = length(coords)

    if C == 0
        @assert nvertices == 1
        simplices = SparseOp{0,C}(sparse([1], [1], [One()], 1, 1))
        return simplices
    end

    if true
        # Use Delaunay.jl

        # Triangulate
        # println("[Calling Delaunay with $nvertices points...]")
        t0 = time_ns()
        mesh = delaunay(S[coords[i][c] for i in axes(coords, 1), c in 1:C])
        # [:Qbb, :Qc, :Qz, :Q12, :QJ]
        t1 = time_ns()
        tdelaunay = round((t1 - t0) / 1.0e9; sigdigits=3)

        # Convert to sparse matrix
        nsimplices = size(mesh.simplices, 1)
        # println("[Delaunay found $nsimplices simplices in $tdelaunay s]")
        @assert size(mesh.simplices, 2) == C + 1
        simplices = MakeSparse{One}(nvertices, nsimplices)
        for j in 1:nsimplices
            for i in @view mesh.simplices[j, :]
                simplices[i, j] = One()
            end
        end
        simplices = SparseOp{0,C}(sparse(simplices))

    else
        # Use MiniQhull.jl

        # Triangulate
        mesh = delaunay(S[coords[i][c] for c in 1:C, i in axes(coords, 1)],
                        "qhull d Qt Qbb Qc Qz")

        # Convert to sparse matrix
        nsimplices = size(mesh, 2)
        @assert size(mesh, 1) == C + 1
        simplices = MakeSparse{One}(nvertices, nsimplices)
        for j in 1:nsimplices
            for i in @view mesh[:, j]
                simplices[i, j] = One()
            end
        end
        # @assert all(i -> 1 ≤ i ≤ nvertices, I)
        # @assert all(j -> 1 ≤ j ≤ nsimplices, J)
        simplices = SparseOp{0,C}(sparse(simplices))
    end

    return simplices
end

export check_delaunay
"""
Check Delaunay condition: No vertex must lie ∈ the circumcentre of a
simplex
"""
function check_delaunay(simplices::SparseOp{0,D,One},
                        lookup::SparseOp{D1,D,One}, lookup1::SparseOp{D,D1,One},
                        coords::Vector{SVector{C,S}},
                        dualcoords::Vector{SVector{C,S}}) where {D,D1,C,S}
    D::Int
    D1::Int
    @assert 0 ≤ D1 ≤ D
    @assert D1 == D - 1
    C::Int

    # This is currently broken because the location of the dual
    # coordinates are not the circumcentres any more. We need to
    # recalculate the circumcentres.
    return

    for i in 1:size(simplices, 2)
        si = sparse_column_rows(simplices, i)
        @assert length(si) == D + 1
        x1i = Form{C,1}(coords[first(si)])
        cci = Form{C,1}(dualcoords[i])
        cri2 = norm2(x1i - cci)
        # Loop over all faces
        for j in sparse_column_rows(lookup, i)
            # Loop over all simplices (except i)
            for k in sparse_column_rows(lookup1, j)
                if k != i
                    # Loop over all vertices
                    for l in sparse_column_rows(simplices, k)
                        # Ignore vertices of simplex i
                        if l ∉ si
                            xl = Form{C,1}(coords[l])
                            d2 = norm2(xl - cci)
                            @assert d2 ≥ cri2 || d2 ≈ cri2
                        end
                    end
                end
            end
        end
    end
end

################################################################################

export refine_coords
"""
Refine a mesh
"""
function refine_coords(oldedges::SparseOp{0,1,One},
                       oldcoords::IDVector{0,SVector{C,S}}) where {C,S}
    noldvertices, noldedges = size(oldedges)
    @assert length(oldcoords) == noldvertices
    nvertices = noldvertices + noldedges
    coords = copy(oldcoords)
    # Loop over all old edges
    for i in axes(oldedges, 2)
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

################################################################################

export improve_mesh
"""
Improve the mesh by moving the circumcentres towards the barycentres
"""
function improve_mesh(simplices::SparseOp{0,R,One},
                      coords::IDVector{0,SVector{C,S}},
                      weights::IDVector{0,S}) where {D,R,C,S}
    C::Int
    @assert 0 ≤ C

    shift_coords = IDVector{0}(zeros(SVector{C,S}, length(coords)))
    shift_weights = IDVector{0}(zeros(S, length(weights)))
    count = 0
    for j in axes(simplices, 2)
        si = sparse_column_rows(simplices, j)
        si = SVector{R + 1}(i for i in si)
        xs = SVector{R + 1}(Form{C,1}(coords[i]) for i in si)
        ws = SVector{R + 1}(Form{C,0}((weights[i],)) for i in si)

        bc = barycentre(xs)
        cc = circumcentre(xs, ws)

        for n in 1:(R + 1)
            i = si[n]
            β = ((xs[n] - bc) ⋅ (cc - bc))[]
            shift_coords[i] -= convert(SVector,
                                       β * (xs[n] - bc) / norm2(xs[n] - bc))
            shift_weights[i] += β
        end
        count += 1
    end
    α = length(shift_weights) / (S(R + 1) * count)
    for i in axes(shift_coords, 1)
        shift_coords[i] *= α
    end
    for i in axes(shift_weights, 1)
        shift_weights[i] *= α
    end

    return shift_coords, shift_weights
end

end
