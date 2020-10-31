module Poisson3d

using DDF
using DifferentialForms
using GLMakie
using IterativeSolvers
using LinearAlgebra
using SparseArrays
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

function main()
    println("Create manifold...")

    D = 3
    S = Float64
    T = Float64

    n = 64
    nelts = Tuple(n for d in 1:D)
    mfd = large_hypercube_manifold(Val(D), S; nelts=nelts, optimize_mesh=true)

    ############################################################################

    # L u == ρ
    # B d u == 0
    # 
    # d u - f == 0
    # δ f == ρ
    # B f == 0

    println("Sample RHS...")

    x₀ = zero(Form{D,1,S}) .+ S(0.25)
    W = S(0.1)
    ρ₀(x) = Form{D,0}((exp(-norm2(x - x₀) / (2 * W^2)),))

    ρ = sample(Fun{D,Pr,0,D,S,T}, ρ₀, mfd)
    u₀ = zero(Fun{D,Pr,0,D,S,T}, mfd)

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

    E = [E0.values N01.values; N10.values E1.values]
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
    A = [N0.values δ.values; -d.values sparse(E1.values)]
    @assert issparse(A)
    b = [ρ.values; n1.values]

    B = [B0.values N01.values; N10.values N1.values]
    @assert issparse(B)
    @assert B * B == B
    c = [u₀.values; n1.values]

    A′ = (E - B) * A + B
    @assert issparse(A′)
    b′ = (E - B) * b + B * c

    ############################################################################

    println("Solve...")

    # x = A′ \ b′

    Pl = Identity()
    # Pl = DiagonalPreconditioner(A′)
    # Pl = AMGPreconditioner{RugeStuben}(A′)
    # Pl = AMGPreconditioner{SmoothedAggregation}(A′)
    # Pl = ilu(A′; τ=0.1)

    t0 = time_ns()

    # x, ch = bicgstabl(A′, b′; Pl=Pl, log=true)
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

    t1 = time_ns()
    tsol = (t1 - t0) / 1.0e+9
    print("\r    iter=$iter time=$(round(tsol; digits=3)) ",
          "res=$(round(solve.residual; sigdigits=3))$(CSI)K")
    println()
    x = solve.x

    # println("    mvps=$(ch.mvps) mtvps=$(ch.mtvps) iters=$(ch.iters) ",
    #         "restart=$(ch.restart) isconverged=$(ch.isconverged)")
    # print("    data={")
    # for (k, v) in ch.data
    #     print("$k=>$v,")
    # end
    # println("}")

    ############################################################################

    println("Analyse solution...")

    uv, fv = vsplit(x, nsimplices(mfd, 0), nsimplices(mfd, 1))
    u = Fun{D,Pr,0,D,S,T}(mfd, uv)
    f = Fun{D,Pr,1,D,S,T}(mfd, fv)

    nsol = norm((E0 - B0) * (laplace(u) - ρ), Inf)
    nbnd = norm(B0 * u - u₀, Inf)
    println("    ‖residual[sol]‖∞=$nsol")
    println("    ‖residual[bnd]‖∞=$nbnd")

    res = (E0 - B0) * (laplace(u) - ρ) + B0 * u - u₀
    nres = norm(res)
    println("    ‖residual‖₂=$nres")

    ############################################################################

    if nsimplices(mfd, 0) ≤ 10000
        println("Plot result...")
        # Can plot ρ, u, res
        plot_function(u, "poisson3d.png")
    end

    ############################################################################

    println("Write result to file...")

    points = [mfd.coords[0][i][d] for d in 1:D, i in 1:nsimplices(mfd, 0)]
    cells = [MeshCell(VTKCellTypes.VTK_TETRA,
                      SVector{D + 1}(i
                                     for i in
                                         sparse_column_rows(mfd.simplices[D],
                                                            j)))
             for j in 1:size(mfd.simplices[D], 2)]
    vtkfile = vtk_grid("poisson3d.vtu", points, cells)

    vtkfile["ρ", VTKPointData()] = ρ.values
    vtkfile["u", VTKPointData()] = u.values
    vtkfile["res", VTKPointData()] = res.values

    vtk_save(vtkfile)

    ############################################################################

    println("Done.")

    return nothing
end

main()

end
