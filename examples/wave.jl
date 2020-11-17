module Wave

# using CairoMakie
# using GLMakie
using DDF
using DiffEqCallbacks
using DifferentialForms
using LinearAlgebra
using Logging: global_logger
using OrdinaryDiffEq
using Printf
using ProgressLogging
using SparseArrays
using StaticArrays
using TerminalLoggers: TerminalLogger

################################################################################

const CSI = "\e["

walltime() = time_ns() / 1.0e+9

global_logger(TerminalLogger())

################################################################################

function rk2(f, y₀, x₀, x₁)
    Δx = x₁ - x₀
    k₁ = f(x₀, y₀)
    k₂ = f(x₀ + Δx / 2, y₀ + Δx / 2 * k₁)
    y₁ = y₀ + Δx * k₂
    return y₁
end

function makeP(mfd::Manifold{D,C,S}) where {D,C,S}
    n = nsimplices(mfd, 0)
    P = MakeSparse{S}(n, n)
    for i in 2:(n - 1)
        P[i, i] = 1
    end
    P[1, 1] = 1 / S(2)
    P[1, n] = 1 / S(2)
    P[n, 1] = 1 / S(2)
    P[n, n] = 1 / S(2)
    P = Op{D,Pr,0,Pr,0}(mfd, sparse(P))
    return P
end

function make_toroidal(A::AbstractSparseMatrix{T,I}) where {T,I}
    nr, nc = size(A)
    # Overwrite rows 1 and nr by shifted versions of rows 2 and nr-1,
    # respectively
    B = MakeSparse{T,I}(nr, nc)
    for c in 1:size(A, 2)
        for (r, v) in sparse_column(A, c)
            if r ∈ (1, nr)
                # ignore this entry
            else
                if r == 2
                    # duplicate this entry
                    c1 = (c == 1 ? nc : c) - 1
                    B[r - 1, c1] = v
                elseif r == nr - 1
                    # duplicate this entry
                    c1 = (c == nc ? 1 : c) + 1
                    B[r + 1, c1] = v
                end
                # copy entry
                B[r, c] = v
            end
        end
    end
    return sparse(B)
end

################################################################################

function wave(::Val{D}, levels::Int) where {D}
    D::Int
    S = Float64
    T = Float64

    println("Create manifold...")

    n = 2^levels
    nelts = Tuple(n for d in 1:D)
    mfd = large_hypercube_manifold(Val(D), S; nelts=nelts, optimize_mesh=false)

    ############################################################################

    println("Sample initial conditions...")

    # TODO: use first-order formulation

    # k = Form{D,1,S}(SVector{D}(d == 1 for d in 1:D))
    # ω = norm(k)
    # function u⁼(t, x)
    #     x = Form{D,1,S}(x)
    #     local u = (sinpi(ω * t - (k ⋅ x)[]) - sinpi(ω * t + (k ⋅ x)[])) / 2
    #     return Form{D,0}((u,))
    # end
    # function ∂ₜu⁼(t, x)
    #     x = Form{D,1,S}(x)
    #     local ∂ₜu = S(π) *
    #                 ω *
    #                 (cospi(ω * t - (k ⋅ x)[]) - cospi(ω * t + (k ⋅ x)[])) / 2
    #     return Form{D,0}((∂ₜu,))
    # end
    # function ∂ₜₜu⁼(t, x)
    #     x = Form{D,1,S}(x)
    #     local ∂ₜₜu = -S(π)^2 *
    #                  ω^2 *
    #                  (sinpi(ω * t - (k ⋅ x)[]) - sinpi(ω * t + (k ⋅ x)[])) / 2
    #     return Form{D,0}((∂ₜₜu,))
    # end

    k = SVector{D}(1 for d in 1:D)
    ω = norm(k)
    function u⁼(t, x)
        local u = cospi(ω * t) * prod(sinpi(k[d] * x[d]) for d in 1:D)
        return Form{D,0}((u,))
    end
    function ∂ₜu⁼(t, x)
        local ∂ₜu = -S(π) *
                    ω *
                    sinpi(ω * t) *
                    prod(sinpi(k[d] * x[d]) for d in 1:D)
        return Form{D,0}((∂ₜu,))
    end

    t₀ = S(0)
    u₀ = sample(Fun{D,Pr,0,D,S,T}, x -> u⁼(t₀, x), mfd)
    ∂ₜu₀ = sample(Fun{D,Pr,0,D,S,T}, x -> ∂ₜu⁼(t₀, x), mfd)

    ############################################################################

    println("Set up equations of motion...")

    # ∂ₜₜu = Δu
    # u̇ := ∂ₜu

    # ∂ₜu = u̇
    # ∂ₜu̇ = Δu

    combine(u, u̇) = [u.values.vec; u̇.values.vec]
    function split(U)
        local n = length(u₀)
        @assert length(U) == 2n
        local uv = U[1:n]
        local u̇v = U[(n + 1):(2n)]
        return (Fun{D,Pr,0,D,S,T}(mfd, IDVector{0}(uv)),
                Fun{D,Pr,0,D,S,T}(mfd, IDVector{0}(u̇v)))
    end

    U₀ = combine(u₀, ∂ₜu₀)

    # Boundary operator for vertices
    N = zero(Op{D,Pr,0,Pr,0}, mfd)
    E = one(Op{D,Pr,0,Pr,0}, mfd)
    B = isboundary(Val(Pr), Val(0), mfd)

    # P = makeP(mfd)
    # @show P

    L = laplace(Val(Pr), Val(0), mfd)

    # B = Op{D,Pr,0,Pr,0}(L.manifold, make_toroidal(B.values))
    # # @show B
    # L = Op{D,Pr,0,Pr,0}(L.manifold, make_toroidal(L.values))
    # # @show L

    function f(U, p, t)
        local u, u̇ = split(U)
        local ∂ₜu = u̇
        local ∂ₜu̇ = L * u
        # local ∂ₜuₜ = sample(Fun{D,Pr,0,D,S,T}, x -> ∂ₜu⁼(t, x), mfd)
        # local ∂ₜₜuₜ = sample(Fun{D,Pr,0,D,S,T}, x -> ∂ₜₜu⁼(t, x), mfd)
        ∂ₜu = (E - B) * ∂ₜu # + B * ∂ₜuₜ
        ∂ₜu̇ = (E - B) * ∂ₜu̇ # + B * ∂ₜₜuₜ
        return combine(∂ₜu, ∂ₜu̇)
    end

    F = [N.values sparse(E.values); L.values N.values]
    F = [(E - B).values N.values; N.values (E - B).values] * F
    f!(∂ₜU, U, p, t) = mul!(∂ₜU, F, U)
    # F = sparse(F')
    # f!(∂ₜU, U, p, t) = mul!(∂ₜU, F', U)

    ############################################################################

    println("Solve...")

    Δx = minimum(get_volumes(mfd, 1))
    # @show Δx

    t₁ = S(1)

    # TODO: Experiment with linear ODE, Hamiltonian systems

    # TODO: Don't use `saveat`, use a `SavingCallback` instead

    # prob = ODEProblem(f, U₀, (t₀, t₁); saveat=(t₁ - t₀) / 8, save_end=false)
    prob = ODEProblem(f!, U₀, (t₀, t₁); saveat=(t₁ - t₀) / 8, save_end=false)

    dt = Δx / 2

    sol = nothing
    nextoutput = walltime()
    @withprogress name = "ODE" begin
        function progress(U, t, integrator)
            if t == t₁ || walltime() ≥ nextoutput
                iter = integrator.iter
                # time = round(integrator.t; sigdigits=3)
                # nsol = round(maximum(abs.(integrator.u)); sigdigits=3)
                # println("    iter=$iter   time=$time ‖sol‖∞=$nsol")
                time = integrator.t
                nsol = norm(integrator.u, Inf)
                @printf "    iter=%d   time=%#.6g   ‖sol‖∞=%#.6g\n" iter time nsol
                @logprogress (t - t₀) / (t₁ - t₀)
                nextoutput = walltime() + 1
            end
        end

        sol = solve(prob, Tsit5(); adaptive=false, dt=dt, alias_u0=true,
                    callback=FunctionCallingCallback(progress))
    end

    # nsteps = 2 * n
    # t = t₀
    # U = U₀
    # @progress name = "RK2" for step in 1:nsteps
    #     tₙ = (nsteps - step) / S(nsteps) * t₀ + step / S(nsteps) * t₁
    #     Uₙ = rk2((t, U) -> f(U, nothing, t), U, t, tₙ)
    #     t = tₙ
    #     U = Uₙ
    # end

    ############################################################################

    println("Analyse solution...")

    # println("Took $(length(sol.t) - 1) time steps")

    # @progress name = "Plotting" for i in axes(sol.t, 1)
    #     # t = (n - i) / S(n) * t₀ + i / S(n) * t₁
    #     # U = sol(t)
    #     t = sol.t[i]
    #     U = sol[i]
    #     u, ∂ₜu = split(U)
    #     uₜ = sample(Fun{D,Pr,0,D,S,T}, x -> u⁼(t, x), mfd)
    #     e = u - uₜ
    #     outdir = "wave$(D)d-$levels"
    #     mkpath(outdir)
    #     plot_function(u, "$outdir/wave$(D)d-u.$(@sprintf "%04d" i - 1).png")
    #     plot_function(e, "$outdir/wave$(D)d-e.$(@sprintf "%04d" i - 1).png")
    # end

    i = last(axes(sol.t, 1))
    t = sol.t[i]
    U = sol[i]
    u, ∂ₜu = split(U)
    uₜ = sample(Fun{D,Pr,0,D,S,T}, x -> u⁼(t, x), mfd)
    e = u - uₜ
    ne = sqrt(norm(e)^2 / length(e))
    println("    ‖u-u⁼‖₂=$ne")

    ############################################################################

    println("Done.")

    return nothing
end

#     levels    error ‖u-u⁼‖₂
#     1         0.02805463824876254
#     2         0.0020292411087007196
#     3         0.00013532326532406398
#     4         8.727862238311897e-6
#     5         5.540717968813277e-7
#     6         3.49003314447833e-8
#     7         2.189781209413498e-9
#     8         1.3712800517004272e-10
#     9         8.580900735370499e-12
#     10        5.369060952547811e-13
#     11        3.1564728009539904e-14
#     12        6.75775317464105e-15
#     13        1.5417520244384893e-14

end
