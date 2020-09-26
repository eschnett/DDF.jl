module DDFPlotting

using AbstractPlotting
using GLMakie
using StaticArrays

using DDF

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

function plot_manifold(filename::String, mfd::Manifold{D,S}) where {D,S}
    D::Int
    @assert D >= 0

    @assert size(mfd.coords, 2) == D

    scene = Scene()

    # Edges
    edges = SVector{D,S}[]
    for i in 1:nsimplices(mfd, 1)
        si = collect(sparse_column_rows(mfd.simplices[1], i))
        @assert length(si) == 2
        xs1 = SVector{D,S}(mfd.coords[si[1], :])
        xs2 = SVector{D,S}(mfd.coords[si[2], :])
        push!(edges, xs1)
        push!(edges, xs2)
    end
    scene = linesegments!(scene, edges, color=:green, linestyle=:solid,
                          linewidth=3)

    # Vertices
    vertices = SVector{D,S}[]
    for i in 1:nsimplices(mfd, 0)
        si = collect(sparse_column_rows(mfd.simplices[0], i))
        @assert length(si) == 1
        xs1 = SVector{D,S}(mfd.coords[si[1], :])
        push!(vertices, xs1)
    end
    scene = scatter!(scene, vertices, strokecolor=:red, strokewidth=5)

    # # Dual edges
    # edges = SVector{D,S}[]
    # for i in 1:nsimplices(mfd, D-1)
    #     si = collect(sparse_column_rows(mfd.simplices[D-1], i))
    #     @assert length(si) == 2
    #     xs1 = SVector{D,S}(mfd.dualcoords[si[1], :])
    #     xs2 = SVector{D,S}(mfd.dualcoords[si[2], :])
    #     push!(edges, xs1)
    #     push!(edges, xs2)
    # end
    # scene = linesegments!(scene, edges, color = :blue, linestyle = :solid,
    #                       linewidth = 3)

    # Dual vertices
    dualvertices = SVector{D,S}[]
    for i in 1:nsimplices(mfd, D)
        # si = collect(sparse_column_rows(mfd.simplices[D], i))
        # @assert length(si) == D + 1
        # xs1 = SVector{D,S}(mfd.dualcoords[si[i], :])
        xs1 = SVector{D,S}(mfd.dualcoords[i, :])
        push!(dualvertices, xs1)
    end
    scene = scatter!(scene, dualvertices, strokecolor=:blue, strokewidth=5)

    save(filename, scene)
    return scene
end

end
