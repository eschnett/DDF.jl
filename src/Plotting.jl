module Plotting

using AbstractPlotting
using AbstractPlotting.MakieLayout
using GLMakie
using LinearAlgebra
using StaticArrays

using ..Manifolds
using ..SparseOps

function plot_example(filename::String)
    D = 3

    coords = SVector{D,Int}[]
    imin = CartesianIndex(ntuple(d -> 0, D))
    imax = CartesianIndex(ntuple(d -> 1, D))
    for i in imin:imax
        push!(coords, SVector{D,Int}(i.I))
    end
    nvertices = length(coords)

    edges = SVector{D,Int}[]
    for i in 1:nvertices, j in (i + 1):nvertices
        push!(edges, coords[i])
        push!(edges, coords[j])
    end

    scene = Scene()
    scene = linesegments!(scene, edges, color=:red, linestyle=:solid,
                          linewidth=3)
    save(filename, scene)
    return
end

export plot_manifold
function plot_manifold(filename::String, mfd::Manifold{D,C,S}) where {D,C,S}
    D::Int
    @assert D ≥ 0

    xmin = SVector{D}(minimum(x -> x[d], mfd.coords[0]) for d in 1:D)
    xmax = SVector{D}(maximum(x -> x[d], mfd.coords[0]) for d in 1:D)
    dx = xmax - xmin
    sz = norm(dx) / 100

    # scene = Scene(resolution=(1024, 1024))
    scene, layout = layoutscene(resolution=(1024, 1024))
    laxis = layout[1, 1] = LAxis(scene, aspect=DataAspect())

    # Edges
    edges = SVector{C,S}[]
    for i in 1:nsimplices(mfd, 1)
        sj = collect(sparse_column_rows(mfd.simplices[1], i))
        @assert length(sj) == 2
        xs1 = mfd.coords[0][sj[1]]
        xs2 = mfd.coords[0][sj[2]]
        push!(edges, xs1)
        push!(edges, xs2)
    end
    linesegments!(laxis, edges, color=:green, linestyle=:solid, linewidth=3)

    # Vertices
    vertices = SVector{C,S}[]
    for i in 1:nsimplices(mfd, 0)
        sj = collect(sparse_column_rows(mfd.simplices[0], i))
        @assert length(sj) == 1
        xs1 = mfd.coords[0][sj[1]]
        push!(vertices, xs1)
    end
    scatter!(laxis, vertices, markersize=sz, strokecolor=:red, strokewidth=5)

    # Dual edges
    dualedges = SVector{C,S}[]
    for i in 1:nsimplices(mfd, D - 1)
        sj = collect(sparse_column_rows(mfd.lookup[(D, D - 1)], i))
        for j in sj
            xs1 = mfd.dualcoords[D - 1][i]
            xs2 = mfd.dualcoords[D][j]
            push!(dualedges, xs1)
            push!(dualedges, xs2)
        end
    end
    linesegments!(laxis, dualedges, color=:red, linestyle=:dash, linewidth=3)

    # Dual vertices
    dualvertices = SVector{C,S}[]
    for i in 1:nsimplices(mfd, D)
        xs1 = mfd.dualcoords[D][i]
        push!(dualvertices, xs1)
    end
    scatter!(laxis, dualvertices, markersize=sz, strokecolor=:blue,
             strokewidth=5)

    xmin -= dx / 10
    xmax += dx / 10
    limits!(laxis, xmin[1], xmax[1], xmin[2], xmax[2])
    save(filename, scene)
    return scene
end

end
