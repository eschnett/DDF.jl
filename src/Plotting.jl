module Plotting

using AbstractPlotting
using AbstractPlotting.MakieLayout
using LinearAlgebra
using StaticArrays

using ..Funs
using ..Manifolds
using ..SparseOps

################################################################################

export plot_manifold
function plot_manifold(mfd::Manifold{D,C,S}, filename=nothing) where {D,C,S}
    D::Int
    C::Int
    @assert 0 ≤ D ≤ C

    xmin = SVector{D}(minimum(x -> x[d], get_coords(mfd)) for d in 1:D)
    xmax = SVector{D}(maximum(x -> x[d], get_coords(mfd)) for d in 1:D)
    dx = xmax - xmin
    sz = norm(dx) / 100

    visible(xs) = true
    # visible(xs) = xs[1] ≥ 0.5 && xs[2] ≤ 0.2 && xs[3] ≥ 0.3

    if C == 2
        scene, layout = layoutscene(resolution=(1024, 1024))
        laxis = layout[1, 1] = LAxis(scene, aspect=DataAspect())
        canvas = laxis
    elseif C == 3
        scene = Scene(resolution=(1024, 1024))
        canvas = scene
    else
        error("C ∉ (2, 3)")
    end

    # Edges
    edges = SVector{C,S}[]
    for i in 1:nsimplices(mfd, 1)
        sj = sparse_column_rows(mfd.simplices[1], i)
        @assert length(sj) == 2
        xs1 = get_coords(mfd)[sj[1]]
        xs2 = get_coords(mfd)[sj[2]]
        if visible(xs1) && visible(xs2)
            push!(edges, xs1)
            push!(edges, xs2)
        end
    end
    linesegments!(canvas, edges, color=:green, linestyle=:solid, linewidth=3)

    # Vertices
    vertices = SVector{C,S}[]
    for i in 1:nsimplices(mfd, 0)
        sj = sparse_column_rows(mfd.simplices[0], i)
        @assert length(sj) == 1
        xs1 = get_coords(mfd)[sj[1]]
        if visible(xs1)
            push!(vertices, xs1)
        end
    end
    scatter!(canvas, vertices, markersize=sz, strokecolor=:red, strokewidth=5)

    # Dual edges
    dualedges = SVector{C,S}[]
    for i in 1:nsimplices(mfd, D - 1)
        sj = sparse_column_rows(lookup(mfd, D, D - 1), i)
        for j in sj
            xs1 = get_dualcoords(mfd, D - 1)[i]
            xs2 = get_dualcoords(mfd, D)[j]
            if visible(xs1) && visible(xs2)
                push!(dualedges, xs1)
                push!(dualedges, xs2)
            end
        end
    end
    linesegments!(canvas, dualedges, color=:red, linestyle=:dash, linewidth=3)

    # Dual vertices
    dualvertices = SVector{C,S}[]
    for i in 1:nsimplices(mfd, D)
        xs1 = get_dualcoords(D, mfd)[i]
        if visible(xs1)
            push!(dualvertices, xs1)
        end
    end
    scatter!(canvas, dualvertices, markersize=sz, strokecolor=:blue,
             strokewidth=5)

    if C == 2
        xmin -= dx / 10
        xmax += dx / 10
        limits!(canvas, xmin[1], xmax[1], xmin[2], xmax[2])
    end

    if filename ≢ nothing
        save(filename, scene)
    end
    return scene
end

################################################################################

export plot_function
function plot_function(fun::Fun{D,P,R,C,S,T},
                       filename=nothing) where {D,P,R,C,S,T}
    D::Int
    C::Int
    @assert 0 ≤ D ≤ C

    mfd = fun.manifold
    xmin = SVector{D}(minimum(x -> x[d], get_coords(mfd)) for d in 1:D)
    xmax = SVector{D}(maximum(x -> x[d], get_coords(mfd)) for d in 1:D)
    dx = xmax - xmin
    sz = norm(dx) / 100

    visible(xs) = true
    # visible(xs) = xs[1] ≥ 0.5 && xs[2] ≤ 0.2 && xs[3] ≥ 0.3

    if C == 2
        scene, layout = layoutscene(resolution=(1024, 1024))
        laxis = layout[1, 1] = LAxis(scene, aspect=DataAspect())
        canvas = laxis
    elseif C == 3
        scene = Scene(resolution=(1024, 1024))
        canvas = scene
    else
        error("C ∉ (2, 3)")
    end

    @assert P == Pr && R == 0
    color = fun.values.vec

    if C == 2
        vertices = [get_coords(mfd)[i][d]
                    for i in axes(get_simplices(mfd, D), 1), d in 1:D]
        connectivity = [Int(sparse_column_rows(get_simplices(mfd, D), j)[n])
                        for j in axes(get_simplices(mfd, D), 2), n in 1:(D + 1)]
        poly!(canvas, vertices, connectivity, color=color,
              strokecolor=(:black, 0.6), strokewidth=4)
    elseif C == 3

        # Edges
        edges = SVector{C,S}[]
        for i in axes(get_simplices(mfd, 1), 2)
            sj = sparse_column_rows(get_simplices(mfd, 1), i)
            @assert length(sj) == 2
            xs1 = get_coords(mfd)[sj[1]]
            xs2 = get_coords(mfd)[sj[2]]
            # if visible(xs1) && visible(xs2)
            push!(edges, xs1)
            push!(edges, xs2)
            # end
        end
        edges::Vector
        linesegments!(canvas, edges, color=:green, linestyle=:solid,
                      linewidth=3)

        # Vertices
        vertices = SVector{C,S}[]
        for i in axes(get_simplices(mfd, 0), 2)
            sj = sparse_column_rows(get_simplices(mfd, 0), i)
            @assert length(sj) == 1
            xs1 = get_coords(mfd)[sj[1]]
            # if visible(xs1)
            push!(vertices, xs1)
            # end
        end
        vertices::Vector
        scatter!(canvas, vertices, markersize=sz, color=color, strokecolor=:red,
                 strokewidth=10)
    end

    if C == 2
        xmin -= dx / 10
        xmax += dx / 10
        limits!(canvas, xmin[1], xmax[1], xmin[2], xmax[2])
    end

    if filename ≢ nothing
        save(filename, scene)
    end
    return scene
end

end
