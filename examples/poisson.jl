module Poisson

using CairoMakie
using DDF
using DifferentialForms
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using StaticArrays
using WriteVTK

const CSI = "\e["

function vsplit(A::Array, sizes...)
    size(A, 1) == sum(sizes) ||
        throw(BoundsError("Excpected: size(A,1) == sum(sizes)"))
    @assert ndims(A) == 1       # TODO
    limits = (0, cumsum(sizes)...)
    # return Tuple((@view A[(limits[i] + 1):limits[i + 1]]) for i in sizes)
    return Tuple(A[(limits[i] + 1):limits[i + 1]] for i in 1:length(sizes))
end

################################################################################

function poisson(::Val{D}, levels::Int) where {D}
    D::Int
    S = Float64
    T = Float64

    println("Create manifold...")

    if D ≤ 2
        mfd = simplex_manifold(Val(D), S; optimize_mesh=false)
        for level in 1:levels
            mfd = refined_manifold(mfd; optimize_mesh=false)
        end
    else
        n = 2^levels
        nelts = Tuple(n for d in 1:D)
        mfd = large_hypercube_manifold(Val(D), S; nelts=nelts,
                                       optimize_mesh=true)
    end

    ############################################################################

    # DA:
    #     ⟨σ|τ⟩ - ⟨u|dτ⟩ = 0
    #     ⟨dσ|v⟩ + ⟨du|dv⟩ + ⟨p|v⟩ = ⟨f|v⟩
    #     ⟨u|q⟩ = 0
    #
    #     ⟨σ|τ⟩ - [u|τ] + ⟨δu|τ⟩ = 0
    #     ⟨dσ|v⟩ + [du|v] - ⟨δdu|v⟩ + ⟨p|v⟩ = ⟨f|v⟩
    #     ⟨u|q⟩ = 0
    #
    #     σ - [u] + δu = 0
    #     dσ + [du] - δdu + p = f
    #     u = 0
    #
    #     σ + δu = 0
    #     dσ + p = f

    # BH:
    #     -⋆f + dp = 0
    #     df       = 0

    # With von Neumann boundary conditions:
    #     L u == ρ
    #     B d u == 0
    #     
    #     d u - f == 0
    #     δ f == ρ
    #     B f == 0

    # With Dirichlet boundary conditions:
    #     L u == ρ
    #     B u == 0
    #     
    #     d u - f == 0
    #     δ f == ρ
    #     B u == 0

    println("Sample RHS...")

    x₀ = zero(Form{D,1,S}) .+ S(0.1)
    σ = S(0.2)
    function ρ⁼(x)
        local r = norm(x - x₀)
        local r′ = r / (sqrt(S(2)) * σ)
        local ρ = 1 / sqrt(2 * S(π) * σ^2)^D * exp(-r′^2)
        return Form{D,0}((ρ,))
    end
    function u⁼(x)
        local r = norm(x - x₀)
        local r′ = r / (sqrt(S(2)) * σ)
        if D == 1
            local u = σ / sqrt(2 * S(π)) * (exp(-r′^2) - 1) + r / 2 * erf(r′)
        elseif D == 2
            if r^8 ≤ eps(S)
                local u = 1 / (4 * S(π)) * r′^2 - 1 / (16 * S(π)) * r′^4 +
                          1 / (72 * S(π)) * r′^6
            else
                local u = -1 / (4 * S(π)) * (-MathConstants.eulergamma +
                           expinti(-r′^2) +
                           log(1 / r′^2))
            end
        elseif D == 3
            local u = -1 / (4 * S(π) * r) * erf(r′)
        elseif D == 4
            if r^8 ≤ eps(S)
                local u = -1 / (8 * S(π)^2 * σ^2) +
                          1 / (16 * S(π)^2 * σ^2) * r′^2 -
                          1 / (48 * S(π)^2 * σ^2) * r′^4 +
                          1 / (192 * S(π)^2 * σ^2) * r′^6
            else
                local u = 1 / (4 * S(π) * r^2) * (exp(-r′^2) - 1)
            end
        else
            @assert false
        end
        return Form{D,0}((u,))
    end

    ρ = sample(Fun{D,Pr,0,D,S,T}, ρ⁼, mfd)
    u₀ = sample(Fun{D,Pr,0,D,S,T}, u⁼, mfd)

    ############################################################################

    println("Determine boundaries...")

    # Boundary operator for vertices
    B0 = isboundary(Val(Pr), Val(0), mfd)

    N0 = zero(Op{D,Pr,0,Pr,0}, mfd)
    N1 = zero(Op{D,Pr,1,Pr,1}, mfd)
    E0 = one(Op{D,Pr,0,Pr,0}, mfd)
    E1 = one(Op{D,Pr,1,Pr,1}, mfd)
    n0 = zero(Fun{D,Pr,0,D,S}, mfd)
    n1 = zero(Fun{D,Pr,1,D,S}, mfd)

    N01 = zero(Op{D,Pr,0,Pr,1}, mfd)
    N10 = zero(Op{D,Pr,1,Pr,0}, mfd)

    E = [
        E0.values N01.values
        N10.values E1.values
    ]
    @assert issparse(E)
    @assert E * E == E

    ############################################################################

    println("Define operators...")

    # We want to solve this:
    #     [0   δ] [u] = [ρ]
    #     [-d  1] [f] = [0]
    #     A x == b
    # but need to handle boundary conditions. The boundary conditions are:
    #     [B  0] [u] = [u₀]
    #     [0  0] [f] = [0 ]
    #     B x == B c
    # Projecting these out of the linear system:
    #     (1-B) A x == (1-B) b
    #     A′ x = b′

    d = deriv(Val(Pr), Val(0), mfd)
    δ = coderiv(Val(Pr), Val(1), mfd)

    # The call to `sparse` is necessary for efficiency
    # <https://github.com/JuliaLang/julia/issues/38209>
    A = [
        N0.values δ.values
        -d.values sparse(E1.values)
    ]
    @assert issparse(A)
    b = [ρ.values.vec; n1.values.vec]

    B = [
        B0.values N01.values
        N10.values N1.values
    ]
    @assert issparse(B)
    @assert B * B == B
    c = [u₀.values.vec; n1.values.vec]

    A′ = (E - B) * A + B
    @assert issparse(A′)
    b′ = (E - B) * b + B * c

    ############################################################################

    println("Solve...")

    if D ≤ 2
        x = A′ \ b′

    else
        t0 = time_ns()

        Pl = Identity()

        x = zeros(T, size(A′, 1))
        solve = bicgstabl_iterator!(x, A′, b′; Pl=Pl)
        lastlog = t0
        iter = 0
        for _ in solve
            iter += 1
            t1 = time_ns()
            tlog = (t1 - lastlog) / 1.0e+9
            if tlog ≥ 1
                lastlog = t1
                tsol = (t1 - t0) / 1.0e+9
                print("\r    iter=$iter time=$(round(tsol; digits=3)) ",
                      "res=$(round(solve.residual; sigdigits=3))$(CSI)K")
            end
        end
        x = solve.x

        t1 = time_ns()
        tsol = (t1 - t0) / 1.0e+9
        print("\r    iter=$iter time=$(round(tsol; digits=3)) ",
              "res=$(round(solve.residual; sigdigits=3))$(CSI)K")
        println()
    end

    ############################################################################

    println("Analyse solution...")

    uv, fv = vsplit(x, nsimplices(mfd, 0), nsimplices(mfd, 1))
    u = Fun{D,Pr,0,D,S,T}(mfd, IDVector{0}(uv))
    f = Fun{D,Pr,1,D,S,T}(mfd, IDVector{1}(fv))

    # nint = norm((E0 - B0) * (laplace(u) - ρ), Inf)
    # nbnd = norm(B0 * (u - u₀), Inf)
    # println("    ‖residual[int]‖∞=$nint")
    # println("    ‖residual[bnd]‖∞=$nbnd")

    r = (E0 - B0) * (laplace(u) - ρ) + B0 * (u - u₀)
    nr = sqrt(norm(r)^2 / length(r))
    println("    ‖residual‖₂=$nr")

    e = u - u₀
    ne = norm(e, 1) / length(e)
    println("    ‖e‖₁=$ne")
    ne = sqrt(norm(e)^2 / length(e))
    println("    ‖e‖₂=$ne")
    ne = norm(e, Inf)
    println("    ‖e‖∞=$ne")

    ############################################################################

    if D ≤ 3
        if nsimplices(mfd, 0) ≤ 10000
            println("Plot result...")
            mkpath("poisson")
            # Can plot ρ, u, u₀, r, e
            plot_function(ρ, "poisson/poisson$(D)d-ρ.png")
            plot_function(u₀, "poisson/poisson$(D)d-u₀.png")
            plot_function(u, "poisson/poisson$(D)d-u.png")
            plot_function(r, "poisson/poisson$(D)d-r.png")
            plot_function(e, "poisson/poisson$(D)d-e.png")
        end
    end

    ############################################################################

    if D ≤ 3
        println("Write result to file...")

        points = [get_coords(mfd)[i][d]
                  for d in 1:D, i in axes(get_simplices(mfd, D), 1)]
        celltype = D == 1 ? VTKCellTypes.VTK_LINE :
                   D == 2 ? VTKCellTypes.VTK_TRIANGLE :
                   D == 3 ? VTKCellTypes.VTK_TETRA : nothing
        cells = [MeshCell(celltype,
                          SVector{D + 1}(Int(i)
                                         for i in
                                             sparse_column_rows(get_simplices(mfd,
                                                                              D),
                                                                j)))
                 for j in axes(get_simplices(mfd, D), 2)]

        mkpath("poisson")
        # append=false, ascii=true
        vtk_grid("poisson/poisson$(D)d", points, cells) do vtkfile
            vtkfile["ρ", VTKPointData()] = ρ.values.vec
            vtkfile["u₀", VTKPointData()] = u₀.values.vec
            vtkfile["u", VTKPointData()] = u.values.vec
            vtkfile["r", VTKPointData()] = r.values.vec
            vtkfile["e", VTKPointData()] = e.values.vec
            return nothing
        end
    end

    ############################################################################

    println("Done.")

    return nothing
end

poisson(::Val{1}) = poisson(Val(1), 5)
poisson(::Val{2}) = poisson(Val(2), 5)
poisson(::Val{3}) = poisson(Val(3), 6)
poisson(::Val{4}) = poisson(Val(4), 4)

poisson(D::Int) = poisson(Val(D))
poisson(D::Int, levels::Int) = poisson(Val(D), levels)

end
