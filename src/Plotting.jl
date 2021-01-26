module Plotting

using AbstractPlotting
using AbstractPlotting.MakieLayout
using ColorSchemes
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
        scene, layout = layoutscene(; resolution=(1024, 1024))
        laxis = layout[1, 1] = LAxis(scene; aspect=DataAspect())
        canvas = laxis
    elseif C == 3
        scene = Scene(; resolution=(1024, 1024))
        canvas = scene
    else
        error("C ∉ (2, 3)")
    end

    # Edges
    edges = SVector{C,S}[]
    for i in axes(get_simplices(mfd, 1), 2)
        sj = sparse_column_rows(get_simplices(mfd, 1), i)
        @assert length(sj) == 2
        xs1 = get_coords(mfd)[sj[1]]
        xs2 = get_coords(mfd)[sj[2]]
        if visible(xs1) && visible(xs2)
            push!(edges, xs1)
            push!(edges, xs2)
        end
    end
    linesegments!(canvas, edges; color=:green, linestyle=:solid, linewidth=3)

    # Vertices
    vertices = SVector{C,S}[]
    for i in axes(get_simplices(mfd, 0), 2)
        sj = sparse_column_rows(get_simplices(mfd, 0), i)
        @assert length(sj) == 1
        xs1 = get_coords(mfd)[sj[1]]
        if visible(xs1)
            push!(vertices, xs1)
        end
    end
    scatter!(canvas, vertices; markersize=sz, strokecolor=:red, strokewidth=5)

    # Dual edges
    dualedges = SVector{C,S}[]
    for i in axes(get_lookup(mfd, D, D - 1), 2)
        sj = sparse_column_rows(get_lookup(mfd, D, D - 1), i)
        for j in sj
            xs1 = get_dualcoords(mfd, D - 1)[i]
            xs2 = get_dualcoords(mfd, D)[j]
            if visible(xs1) && visible(xs2)
                push!(dualedges, xs1)
                push!(dualedges, xs2)
            end
        end
    end
    linesegments!(canvas, dualedges; color=:red, linestyle=:dash, linewidth=3)

    # Dual vertices
    dualvertices = SVector{C,S}[]
    for i in axes(get_dualcoords(mfd, D), 1)
        xs1 = get_dualcoords(mfd, D)[i]
        if visible(xs1)
            push!(dualvertices, xs1)
        end
    end
    scatter!(canvas, dualvertices; markersize=sz, strokecolor=:blue,
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
export plot_function1d

function plot_function(fun::Fun{D,P,R,1,S,T},
                       filename=nothing) where {D,P,R,S,T}
    D::Int
    C = 1
    @assert 0 ≤ D ≤ C

    mfd = fun.manifold

    visible(xs) = true
    # visible(xs) = xs[1] ≥ 0.5 && xs[2] ≤ 0.2 && xs[3] ≥ 0.3

    scene, layout = layoutscene(; resolution=(1024, 1024))
    laxis = layout[1, 1] = LAxis(scene)
    canvas = laxis

    @assert P == Pr && R == 0

    points = [(get_coords(mfd)[i][1], fun.values[i])
              for i in axes(get_simplices(mfd, D), 1)]
    sort!(points)
    lines!(canvas, points; color=(:black, 0.6), linewidth=4)
    scatter!(canvas, points; markersize=4, strokecolor=:red, strokewidth=4)

    # xmin = SVector{D}(minimum(x -> x[d], get_coords(mfd)) for d in 1:D)
    # xmax = SVector{D}(maximum(x -> x[d], get_coords(mfd)) for d in 1:D)
    # dx = xmax - xmin
    # xmin -= dx / 10
    # xmax += dx / 10
    # limits!(canvas, xmin[1], xmax[1], xmin[2], xmax[2])

    if filename ≢ nothing
        save(filename, scene)
    end

    return scene
end

function plot_function1d(fun::Fun{D,P,R,2,S,T},
                         filename=nothing) where {D,P,R,S,T}
    D::Int
    C = 2
    @assert 0 ≤ R ≤ D ≤ C

    mfd = fun.manifold

    # visible(xs) = true
    visible(xs) = 0.5 - 0.001 ≤ xs[2] ≤ 0.5 + 0.001

    scene, layout = layoutscene(; resolution=(1024, 1024))
    laxis = layout[1, 1] = LAxis(scene)
    canvas = laxis

    @assert P == Pr

    if R == 0
        points = [(get_coords(mfd, R)[i][1],
                   fun.values[i] / get_volumes(mfd, R)[i])
                  for i in axes(get_simplices(mfd, R), 2)
                  if visible(get_coords(mfd, R)[i])]
        sort!(points)
        lines!(canvas, points; color=(:black, 0.6), linewidth=4)
        scatter!(canvas, points; markersize=4, strokecolor=:red, strokewidth=4)

        points = [(get_coords(mfd, R)[i][1],
                   fun.values[i] / get_volumes(mfd, R)[i])
                  for i in axes(get_simplices(mfd, R), 2)
                  if visible(get_coords(mfd, R)[i] - 1 / 8)]
        sort!(points)
        lines!(canvas, points; color=(:black, 0.6), linewidth=4)
        scatter!(canvas, points; markersize=4, strokecolor=:red, strokewidth=4)

        points = [(get_coords(mfd, R)[i][1],
                   fun.values[i] / get_volumes(mfd, R)[i])
                  for i in axes(get_simplices(mfd, R), 2)
                  if visible(get_coords(mfd, R)[i] - 1 / 16)]
        sort!(points)
        lines!(canvas, points; color=(:blue, 0.6), linewidth=4)
        scatter!(canvas, points; markersize=4, strokecolor=:purple,
                 strokewidth=4)
    elseif R == 1
        points = [(get_coords(mfd, R)[i][1],
                   fun.values[i] *
                   (get_coords(mfd, 0)[sparse_column_rows(get_simplices(mfd, R),
                                                          i)[2]] -
                    get_coords(mfd, 0)[sparse_column_rows(get_simplices(mfd, R),
                                                          i)[1]]) ⋅ (1, 0))
                  for i in axes(get_simplices(mfd, R), 2)
                  if visible(get_coords(mfd, R)[i])]
        sort!(points)
        lines!(canvas, points; color=(:black, 0.6), linewidth=4)
        scatter!(canvas, points; markersize=4, strokecolor=:red, strokewidth=4)

        points = [(get_coords(mfd, R)[i][1],
                   fun.values[i] *
                   (get_coords(mfd, 0)[sparse_column_rows(get_simplices(mfd, R),
                                                          i)[2]] -
                    get_coords(mfd, 0)[sparse_column_rows(get_simplices(mfd, R),
                                                          i)[1]]) ⋅ (1, 0))
                  for i in axes(get_simplices(mfd, R), 2)
                  if visible(get_coords(mfd, R)[i] - 1 / 16)]
        sort!(points)
        lines!(canvas, points; color=(:black, 0.6), linewidth=4)
        scatter!(canvas, points; markersize=4, strokecolor=:red, strokewidth=4)

        points = [(get_coords(mfd, R)[i][1],
                   fun.values[i] *
                   (get_coords(mfd, 0)[sparse_column_rows(get_simplices(mfd, R),
                                                          i)[2]] -
                    get_coords(mfd, 0)[sparse_column_rows(get_simplices(mfd, R),
                                                          i)[1]]) ⋅ (1, 0))
                  for i in axes(get_simplices(mfd, R), 2)
                  if visible(get_coords(mfd, R)[i] - 1 / 32)]
        sort!(points)
        lines!(canvas, points; color=(:blue, 0.6), linewidth=4)
        scatter!(canvas, points; markersize=4, strokecolor=:purple,
                 strokewidth=4)
    end

    # xmin = SVector{D}(minimum(x -> x[d], get_coords(mfd)) for d in 1:D)
    # xmax = SVector{D}(maximum(x -> x[d], get_coords(mfd)) for d in 1:D)
    # dx = xmax - xmin
    # xmin -= dx / 10
    # xmax += dx / 10
    # limits!(canvas, xmin[1], xmax[1], xmin[2], xmax[2])

    if filename ≢ nothing
        save(filename, scene)
    end

    return scene
end

function plot_function(fun::Fun{D,P,R,2,S,T},
                       filename=nothing) where {D,P,R,S,T}
    D::Int
    C = 2
    @assert 0 ≤ D ≤ C

    mfd = fun.manifold

    visible(xs) = true
    # visible(xs) = xs[1] ≥ 0.5 && xs[2] ≤ 0.2 && xs[3] ≥ 0.3

    scene, layout = layoutscene(; resolution=(1024, 1024))
    laxis = layout[1, 1] = LAxis(scene; aspect=DataAspect())
    canvas = laxis

    @assert P == Pr && R == 0

    color = fun.values.vec
    colormap = ColorSchemes.plasma.colors
    colorrange = (minimum(color), maximum(color))

    vertices = [get_coords(mfd)[i][d]
                for i in axes(get_simplices(mfd, D), 1), d in 1:D]
    connectivity = [Int(sparse_column_rows(get_simplices(mfd, D), j)[n])
                    for j in axes(get_simplices(mfd, D), 2), n in 1:(D + 1)]
    poly!(canvas, vertices, connectivity; color=color, colormap=colormap,
          colorrange=colorrange, strokecolor=(:black, 0.6), strokewidth=4)
    colorlegend!(colormap, colorrange)

    xmin = SVector{D}(minimum(x -> x[d], get_coords(mfd)) for d in 1:D)
    xmax = SVector{D}(maximum(x -> x[d], get_coords(mfd)) for d in 1:D)
    dx = xmax - xmin
    xmin -= dx / 10
    xmax += dx / 10
    limits!(canvas, xmin[1], xmax[1], xmin[2], xmax[2])

    if filename ≢ nothing
        save(filename, scene)
    end

    return scene
end

function plot_function(fun::Fun{D,P,R,3,S,T},
                       filename=nothing) where {D,P,R,S,T}
    D::Int
    C = 3
    @assert 0 ≤ D ≤ C

    mfd = fun.manifold

    visible(xs) = true
    # visible(xs) = xs[1] ≥ 0.5 && xs[2] ≤ 0.2 && xs[3] ≥ 0.3

    xmin = SVector{D}(minimum(x -> x[d], get_coords(mfd)) for d in 1:D)
    xmax = SVector{D}(maximum(x -> x[d], get_coords(mfd)) for d in 1:D)
    dx = xmax - xmin
    sz = norm(dx) / 100

    scene = Scene(; resolution=(1024, 1024))
    canvas = scene

    @assert P == Pr && R == 0

    color = fun.values.vec

    # Edges
    edges = SVector{C,S}[]
    for i in axes(get_simplices(mfd, 1), 2)
        sj = sparse_column_rows(get_simplices(mfd, 1), i)
        @assert length(sj) == 2
        xs1 = get_coords(mfd)[sj[1]]
        xs2 = get_coords(mfd)[sj[2]]
        if visible(xs1) && visible(xs2)
            push!(edges, xs1)
            push!(edges, xs2)
        end
    end
    linesegments!(canvas, edges; color=:green, linestyle=:solid, linewidth=3)

    # Vertices
    vertices = SVector{C,S}[]
    for i in axes(get_simplices(mfd, 0), 2)
        sj = sparse_column_rows(get_simplices(mfd, 0), i)
        @assert length(sj) == 1
        xs1 = get_coords(mfd)[sj[1]]
        if visible(xs1)
            push!(vertices, xs1)
        end
    end
    scatter!(canvas, vertices; markersize=sz, color=color, strokecolor=:red,
             strokewidth=10)

    if filename ≢ nothing
        save(filename, scene)
    end

    return scene
end

end
