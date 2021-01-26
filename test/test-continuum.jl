using DDF

using Bernstein
using ComputedFieldTypes
using DifferentialForms
using LinearAlgebra
using StaticArrays

@testset "Barycentric basis functions D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P
    P == Pr || continue

    N = D + 1
    S = Float64
    mfd = simplex_manifold(Val(D), S)
    d = R == 0 ? nothing : deriv(Val(P), Val(R - 1), mfd)

    e = Form{D + 1,0}((1,))

    λs = [SVector{N,S}(i.I) / sum(i.I)
          for i in
              CartesianIndex(ntuple(d -> 0, N)):CartesianIndex(ntuple(d -> 1,
                                                                      N))
          if sum(i.I) > 0]
    λs::Vector{SVector{N,S}}
    @assert all(sum(λ) ≈ 1 for λ in λs)

    basis_λ = Continuum.basis_λ

    if D == 0
        # vertices: 1

        e1 = Form{D + 1,1}((1,))

        if R == 0
            bs = [λ -> λ[1] * e]
        end
    elseif D == 1
        # vertices: 1 2
        # edges:    (1,2)
        @test collect(sparse_column_rows(get_simplices(mfd, 0), ID{0}(1))) ==
              [ID{0}(1)]
        @test collect(sparse_column_rows(get_simplices(mfd, 0), ID{0}(2))) ==
              [ID{0}(2)]
        @test collect(sparse_column_rows(get_simplices(mfd, 1), ID{1}(1))) ==
              [ID{0}(1), ID{0}(2)]

        e1 = Form{D + 1,1}((1, 0))
        e2 = Form{D + 1,1}((0, 1))

        if R == 0
            e = Form{D + 1,R}((1,))
            bs = [λ -> λ[1] * e, λ -> λ[2] * e]
        elseif R == 1
            @test d.values == [-1 1]
            e1 = Form{D + 1,R}((1, 0))
            e2 = Form{D + 1,R}((0, 1))
            bs = [λ -> λ[1] * e2 - λ[2] * e1]
        end
    elseif D == 2
        # vertices: 1 2 3
        # edges:    (1,2) (1,3) (2,3)
        # faces:    (1,2,3)
        @test collect(sparse_column_rows(get_simplices(mfd, 0), ID{0}(1))) ==
              [ID{0}(1)]
        @test collect(sparse_column_rows(get_simplices(mfd, 0), ID{0}(2))) ==
              [ID{0}(2)]
        @test collect(sparse_column_rows(get_simplices(mfd, 0), ID{0}(3))) ==
              [ID{0}(3)]
        @test collect(sparse_column_rows(get_simplices(mfd, 1), ID{1}(1))) ==
              [ID{0}(1), ID{0}(2)]
        @test collect(sparse_column_rows(get_simplices(mfd, 1), ID{1}(2))) ==
              [ID{0}(1), ID{0}(3)]
        @test collect(sparse_column_rows(get_simplices(mfd, 1), ID{1}(3))) ==
              [ID{0}(2), ID{0}(3)]
        @test collect(sparse_column_rows(get_simplices(mfd, 2), ID{2}(1))) ==
              [ID{0}(1), ID{0}(2), ID{0}(3)]

        e1 = Form{D + 1,1}((1, 0, 0))
        e2 = Form{D + 1,1}((0, 1, 0))
        e3 = Form{D + 1,1}((0, 0, 1))

        if R == 0
            bs = [λ -> λ[1] * e, λ -> λ[2] * e, λ -> λ[3] * e]
        elseif R == 1
            @test d.values == [
                -1 1 0
                -1 0 1
                0 -1 1
            ]
            bs = [λ -> λ[1] * e2 - λ[2] * e1, λ -> λ[1] * e3 - λ[3] * e1,
                  λ -> λ[2] * e3 - λ[3] * e2]
        elseif R == 2
            @test d.values == [1 -1 1]
            e12 = e1 ∧ e2
            e13 = e1 ∧ e3
            e23 = e2 ∧ e3
            bs = [λ -> λ[1] * e23 - λ[2] * e13 + λ[3] * e12]
        end
    else
        bs = []
    end

    for n in 1:length(bs), λ in λs
        @test basis_λ(Form{D,R}, n, λ) == bs[n](λ)
    end
end

@testset "Cartesian basis functions D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P
    P == Pr || continue

    S = Float64
    mfd = orthogonal_simplex_manifold(Val(D), S; optimize_mesh=false)

    e = Form{D,0}((1,))

    xs = SVector{D + 1,SVector{D,S}}(get_coords(mfd)[i]
                                     for i in ID{0}(1):ID{0}(D + 1))
    x2λ = cartesian2barycentric_setup(xs)
    dλ2dx = Continuum.dbarycentric2dcartesian_setup(Form{D,R}, xs)

    if D == 0
        pts = [SVector{D,S}()]
    else
        pts = [sum(i.I .* xs) / sum(i.I)
               for i in
                   CartesianIndex(ntuple(d -> 0, D + 1)):CartesianIndex(ntuple(d -> 1,
                                                                               D +
                                                                               1))
               if sum(i.I) > 0]
    end
    pts::Vector{SVector{D,S}}

    basis_x = Continuum.basis_x

    # Test basis functions at various points
    if D == 0
        if R == 0
            bs = [x -> e]
        end
    elseif D == 1
        e1 = Form{D,1}((1,))
        if R == 0
            bs = [x -> (1 - x[1]) * e, x -> x[1] * e]
        elseif R == 1
            bs = [x -> e1]
        end
    elseif D == 2
        e1 = Form{D,1}((1, 0))
        e2 = Form{D,1}((0, 1))
        if R == 0
            e = Form{D,R,S}((1,))
            bs = [x -> (1 - x[1] - x[2]) * e, x -> x[1] * e, x -> x[2] * e]
        elseif R == 1
            bs = [x -> -(x[2] - 1) * e1 + x[1] * e2,
                  x -> x[2] * e1 - (x[1] - 1) * e2, x -> -x[2] * e1 + x[1] * e2]
        elseif R == 2
            e12 = e1 ∧ e2
            bs = [x -> e12]
        end
    else
        bs = []
    end

    for n in 1:length(bs), x in pts
        @test basis_x(Form{D,R}, x2λ, dλ2dx, n, x) ≈ bs[n](x)
    end
end

# @DISABLED @testset "Cartesian basis functions D=$D P=$P R=$R" for D in
#                                                                          0:Dmax,
# P in (Pr, Dl),
# R in 0:D
# 
#     # TODO: all P
#     P == Pr || continue
# 
#     S = Float64
#     mfd = simplex_manifold(Val(D), S)
# 
#     xs = SVector{D + 1,SVector{D,S}}(get_coords(mfd)[i]
#                                      for i in ID{0}(1):ID{0}(D + 1))
#     x2λ = cartesian2barycentric_setup(xs)
#     dλ2dx = Continuum.dbarycentric2dcartesian_setup(Form{D,R}, xs)
# 
#     VR = IDVector{R}
#     I0(is...) = map(ID{0}, is)
# 
#     for n in axes(get_simplices(mfd, P == Pr ? R : D - R), 2)
#         for i in axes(get_simplices(mfd, 0), 1)
#             x = get_coords(mfd)[i]
#             bx = Continuum.basis_x(Form{D,R}, x2λ, dλ2dx, Int(n), x)
#             if R == 0
#                 @test bx ≈
#                       Form{D,R}((Int(i) == Int(n) ? get_volumes(mfd, R)[n] : 0,))
#             elseif R == D
#                 if !(bx ≈ Form{D,R}((get_volumes(mfd, R)[n],)))
#                     @show D R n i bx
#                     @show get_volumes(mfd, R)
#                 end
#                 @test bx ≈ Form{D,R}((get_volumes(mfd, R)[n],))
#             elseif D == 2 && R == 1
#                 v1, v2 = VR([I0(1, 2), I0(1, 3), I0(2, 3)])[n]
#                 dx = get_coords(mfd)[v2] - get_coords(mfd)[v1]
#                 #TODO WRONG ORDER OF BASIS FUNCTIONS
#                 if !(bx ≈ Form{D,R}((i in (v1, v2)) * dx))
#                     @show D R n i
#                     for m in axes(get_simplices(mfd, P == Pr ? R : D - R), 2)
#                         v1, v2 = VR([I0(1, 2), I0(1, 3), I0(2, 3)])[m]
#                         dx = get_coords(mfd)[v2] - get_coords(mfd)[v1]
#                         @show Form{D,R}((i in (v1, v2)) * dx)
#                     end
#                 end
#                 v1, v2 = VR([I0(1, 2), I0(1, 3), I0(2, 3)])[n]
#                 dx = get_coords(mfd)[v2] - get_coords(mfd)[v1]
#                 @test bx ≈ Form{D,R}((i in (v1, v2)) * dx)
#             elseif D == 3 && R == 1
#                 v1, v2 = VR([I0(1, 2), I0(1, 3), I0(2, 3), I0(1, 4), I0(2, 4),
#                              I0(3, 4)])[n]
#                 dx = get_coords(mfd)[v2] - get_coords(mfd)[v1]
#                 @test bx ≈ Form{D,R}((i in (v1, v2)) * dx)
#             elseif D == 4 && R == 1
#                 v1, v2 = VR([I0(1, 2), I0(1, 3), I0(2, 3), I0(1, 4), I0(2, 4),
#                              I0(3, 4), I0(1, 5), I0(2, 5), I0(3, 5), I0(4, 5)])[n]
#                 dx = get_coords(mfd)[v2] - get_coords(mfd)[v1]
#                 @test bx ≈ Form{D,R}((i in (v1, v2)) * dx)
#             elseif D == 5 && R == 1
#                 v1, v2 = VR([I0(1, 2), I0(1, 3), I0(2, 3), I0(1, 4), I0(2, 4),
#                              I0(3, 4), I0(1, 5), I0(2, 5), I0(3, 5), I0(4, 5),
#                              I0(1, 6), I0(2, 6), I0(3, 6), I0(4, 6), I0(5, 6)])[n]
#                 dx = Form{D,R}(get_coords(mfd)[v2] - get_coords(mfd)[v1])
#                 @test bx ≈ (i in (v1, v2)) * dx
#             elseif D == 3 && R == 2
#                 v1, v2, v3 = VR([I0(1, 2, 3), I0(1, 2, 4), I0(1, 3, 4),
#                                  I0(2, 3, 4)])[n]
#                 dx = Form{D,1}(get_coords(mfd)[v2] - get_coords(mfd)[v1]) ∧
#                      Form{D,1}(get_coords(mfd)[v3] - get_coords(mfd)[v1]) / 2
#                 @test bx ≈ (i in (v1, v2, v3)) * dx
#             elseif D == 4 && R == 2
#                 v1, v2, v3 = VR([I0(1, 2, 3), I0(1, 2, 4), I0(1, 3, 4),
#                                  I0(2, 3, 4), I0(1, 2, 5), I0(1, 3, 5),
#                                  I0(2, 3, 5), I0(1, 4, 5), I0(2, 4, 5),
#                                  I0(3, 4, 5)])[n]
#                 dx = Form{D,1}(get_coords(mfd)[v2] - get_coords(mfd)[v1]) ∧
#                      Form{D,1}(get_coords(mfd)[v3] - get_coords(mfd)[v1]) / 2
#                 @test bx ≈ (i in (v1, v2, v3)) * dx
#             elseif D == 5 && R == 2
#                 v1, v2, v3 = VR([I0(1, 2, 3), I0(1, 2, 4), I0(1, 3, 4),
#                                  I0(2, 3, 4), I0(1, 2, 5), I0(1, 3, 5),
#                                  I0(2, 3, 5), I0(1, 4, 5), I0(2, 4, 5),
#                                  I0(3, 4, 5), I0(1, 2, 6), I0(1, 3, 6),
#                                  I0(2, 3, 6), I0(1, 4, 6), I0(2, 4, 6),
#                                  I0(3, 4, 6), I0(1, 5, 6), I0(2, 5, 6),
#                                  I0(3, 5, 6), I0(4, 5, 6)])[n]
#                 dx = Form{D,1}(get_coords(mfd)[v2] - get_coords(mfd)[v1]) ∧
#                      Form{D,1}(get_coords(mfd)[v3] - get_coords(mfd)[v1]) / 2
#                 @test bx ≈ (i in (v1, v2, v3)) * dx
#             elseif D == 4 && R == 3
#                 v1, v2, v3, v4 = VR([I0(1, 2, 3, 4), I0(1, 2, 3, 5),
#                                      I0(1, 2, 4, 5), I0(1, 3, 4, 5),
#                                      I0(2, 3, 4, 5)])[n]
#                 dx = Form{D,1}(get_coords(mfd)[v2] - get_coords(mfd)[v1]) ∧
#                      Form{D,1}(get_coords(mfd)[v3] - get_coords(mfd)[v1]) ∧
#                      Form{D,1}(get_coords(mfd)[v4] - get_coords(mfd)[v1]) / 6
#                 @test bx ≈ (i in (v1, v2, v3, v4)) * dx
#             end
#         end
#     end
# end

@testset "Evaluate a function D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P
    P == Pr || continue

    S = Float64
    mfd = simplex_manifold(Val(D), S)

    T = S

    f0 = zero(Fun{D,P,R,D,S,T}, mfd)
    nvalues = nsimplices(mfd, P == Pr ? R : D - R)
    fi = Fun{D,P,R,D,S,T}(mfd, IDVector{R}([T(i) for i in 1:nvalues]))
    fx = Fun{D,P,R,D,S,SVector{D,T}}(mfd, get_coords(mfd, R))
    f1 = Fun{D,P,R,D,S,T}(mfd, IDVector{R}(rand(T, nvalues)))
    f2 = Fun{D,P,R,D,S,T}(mfd, IDVector{R}(rand(T, nvalues)))
    a = rand(T)
    b = rand(T)

    if R == 0
        for (i, x) in enumerate(get_coords(mfd, R))
            @test evaluate(fi, x) ≈ Form{D,R}((T(i),))
            @test evaluate(fx, x) ≈ Form{D,R}((x,))
        end
    end

    xs = [
        collect(get_coords(mfd, R))
        [random_point(Val(D), mfd) for i in 1:10]
    ]
    for x in xs
        @test evaluate(f0, x) == zero(Form{D,R,T})
        @test evaluate(f1 + f2, x) ≈ evaluate(f1, x) + evaluate(f2, x)
        @test evaluate(a * f1, x) ≈ a * evaluate(f1, x)
    end

    if D == 0
        # vertices: 1
        x1 = get_coords(mfd)[ID{0}(1)]
        fi1 = evaluate(fi, x1)
        if R == 0
            @test fi1 ≈ Form{D,0}((1,))
        else
            @assert false
        end
    elseif D == 1
        # vertices: 1 2
        # edges:    (1,2)
        x1 = get_coords(mfd)[ID{0}(1)]
        x2 = get_coords(mfd)[ID{0}(2)]
        fi1 = evaluate(fi, x1)
        fi2 = evaluate(fi, x2)
        if R == 0
            @test fi1 ≈ Form{D,0}((1,))
            @test fi2 ≈ Form{D,0}((2,))
        elseif R == 1
            @test fi1 ≈ Form{D,1}((1,))
            @test fi2 ≈ Form{D,1}((1,))
        else
            @assert false
        end
    elseif D == 2
        # vertices: 1 2 3
        # edges:    (1,2) (1,3) (2,3)
        # faces:    (1,2,3)
        x1 = get_coords(mfd)[ID{0}(1)]
        x2 = get_coords(mfd)[ID{0}(2)]
        x3 = get_coords(mfd)[ID{0}(3)]
        fi1 = evaluate(fi, x1)
        fi2 = evaluate(fi, x2)
        fi3 = evaluate(fi, x3)
        if R == 0
            b1 = Form{D,0}((S(1),))
            b2 = Form{D,0}((S(1),))
            b3 = Form{D,0}((S(1),))
            @test fi1 ≈ 1 * b1
            @test fi2 ≈ 2 * b2
            @test fi3 ≈ 3 * b3
        elseif R == 1
            b1 = Form{D,1}(x2 - x1)
            b2 = Form{D,1}(x3 - x1)
            b3 = Form{D,1}(x3 - x2)
            @test fi1 ≈ 1 * b1 + 2 * b2
            @test fi2 ≈ 1 * b1 + 3 * b3
            @test fi3 ≈ 2 * b2 + 3 * b3
        elseif R == 2
            b1 = Form{D,1}(x2 - x1) ∧ Form{D,1}(x3 - x1) / 2
            @test fi1 ≈ 1 * b1
            @test fi2 ≈ 1 * b1
            @test fi3 ≈ 1 * b1
        else
            @assert false
        end
    end

    if R < D
        dfi = deriv(fi)
        if D == 0
            @assert false
        elseif D == 1
            # vertices: 1 2
            # edges:    (1,2)
            x1 = get_coords(mfd)[ID{0}(1)]
            x2 = get_coords(mfd)[ID{0}(2)]
            fi1 = evaluate(fi, x1)
            fi2 = evaluate(fi, x2)
            dfi1 = evaluate(dfi, x1)
            dfi2 = evaluate(dfi, x2)
            if R == 0
                @test fi1 ≈ Form{D,0}((1,))
                @test fi2 ≈ Form{D,0}((2,))
                @test dfi1 ≈ (-1 + 2) * Form{D,1}(-x1 + x2)
                @test dfi2 ≈ (-1 + 2) * Form{D,1}(-x1 + x2)
            else
                @assert false
            end
        elseif D == 2
            # vertices: 1 2 3
            # edges:    (1,2) (1,3) (2,3)
            # faces:    (1,2,3)
            x1 = get_coords(mfd)[ID{0}(1)]
            x2 = get_coords(mfd)[ID{0}(2)]
            x3 = get_coords(mfd)[ID{0}(3)]
            fi1 = evaluate(fi, x1)
            fi2 = evaluate(fi, x2)
            fi3 = evaluate(fi, x3)
            dfi1 = evaluate(dfi, x1)
            dfi2 = evaluate(dfi, x2)
            dfi3 = evaluate(dfi, x3)
            if R == 0
                b1 = Form{D,0}((S(1),))
                b2 = Form{D,0}((S(1),))
                b3 = Form{D,0}((S(1),))
                @test fi1 ≈ 1 * b1
                @test fi2 ≈ 2 * b2
                @test fi3 ≈ 3 * b3
                db1 = Form{D,1}(x2 - x1)
                db2 = Form{D,1}(x3 - x1)
                db3 = Form{D,1}(x3 - x2)
                @test dfi1 ≈ (-1 + 2) * db1 + (-1 + 3) * db2
                @test dfi2 ≈ (-1 + 2) * db1 + (-2 + 3) * db3
                @test dfi3 ≈ (-1 + 3) * db2 + (-2 + 3) * db3
            elseif R == 1
                b1 = Form{D,1}(x2 - x1)
                b2 = Form{D,1}(x3 - x1)
                b3 = Form{D,1}(x3 - x2)
                @test fi1 ≈ 1 * b1 + 2 * Form{D,1}(x3 - x1)
                @test fi2 ≈ 1 * b1 + 3 * Form{D,1}(x3 - x2)
                @test fi3 ≈ 2 * b2 + 3 * Form{D,1}(x3 - x2)
                db1 = Form{D,1}(x2 - x1) ∧ Form{D,1}(x3 - x1) / 2
                @test dfi1 ≈ (1 - 2 + 3) * db1
                @test dfi2 ≈ (1 - 2 + 3) * db1
                @test dfi3 ≈ (1 - 2 + 3) * db1
            else
                @assert false
            end
        end
    end
end

@testset "Sample a function D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P
    P == Pr || continue

    # TODO: all R
    R == 0 || continue

    S = Float64
    # mfd = simplex_manifold(Val(D), S)
    # for level in 1:2
    #     mfd = refined_manifold(mfd; optimize_mesh=false)
    # end
    mfd = large_hypercube_manifold(Val(D), S; nelts=ntuple(d -> 2, D),
                                   optimize_mesh=false)

    # nomenclature: fDRN where D=dim, R=rank, N=basis element
    # D = 0, R = 0
    c₀₀₁ = rand(S)
    f₀₀₁(x) = c₀₀₁ * Form{0,0,S}((1,)) # 1
    f₀₀(x) = f₀₀₁(x)
    # D = 1, R = 0
    c₁₀₁ = rand(S)
    f₁₀₁(x) = c₁₀₁ * Form{1,0,S}((1,)) # 1
    c₁₀₂ = rand(S)
    f₁₀₂(x) = c₁₀₂ * x[1] * Form{1,0,S}((1,)) # x
    f₁₀(x) = f₁₀₁(x) + f₁₀₂(x)
    # D = 1, R = 1
    c₁₁₁ = rand(S)
    f₁₁₁(x) = c₁₁₁ * Form{1,1,S}((1,)) # dx
    f₁₁(x) = f₁₁₁(x)
    # D = 2, R = 0
    c₂₀₁ = rand(S)
    f₂₀₁(x) = c₂₀₁ * Form{2,0,S}((1,)) # 1
    c₂₀₂ = rand(S)
    f₂₀₂(x) = c₂₀₂ * x[1] * Form{2,0,S}((1,)) # x
    c₂₀₃ = rand(S)
    f₂₀₃(x) = c₂₀₃ * x[2] * Form{2,0,S}((1,)) # y
    f₂₀(x) = f₂₀₁(x) + f₂₀₂(x) + f₂₀₃(x)
    # D = 2, R = 1
    c₂₁₁ = S(1) # rand(S)
    f₂₁₁(x) = c₂₁₁ * Form{2,1,S}((1, 0)) # dx
    c₂₁₂ = S(0) # rand(S)
    f₂₁₂(x) = c₂₁₂ * Form{2,1,S}((0, 1)) # dy
    c₂₁₃ = S(0) # rand(S)
    f₂₁₃(x) = c₂₁₃ * (-x[2] * Form{2,1,S}((1, 0)) + x[1] * Form{2,1,S}((0, 1))) # - y dx + x dy
    f₂₁(x) = f₂₁₁(x) + f₂₁₂(x) + f₂₁₃(x)
    # D = 2, R = 2
    c₂₂₁ = rand(S)
    f₂₂₁(x) = c₂₂₁ * Form{2,2,S}((1, 1)) # dx ∧ dy
    f₂₂(x) = f₂₂₁(x)
    # D = 3, R = 0
    c₃₀₁ = rand(S)
    f₃₀₁(x) = c₃₀₁ * Form{3,0,S}((1,)) # 1
    c₃₀₂ = rand(S)
    f₃₀₂(x) = c₃₀₂ * x[1] * Form{3,0,S}((1,)) # x
    c₃₀₃ = rand(S)
    f₃₀₃(x) = c₃₀₃ * x[2] * Form{3,0,S}((1,)) # y
    c₃₀₄ = rand(S)
    f₃₀₄(x) = c₃₀₄ * x[3] * Form{3,0,S}((1,)) # z
    f₃₀(x) = f₃₀₁(x) + f₃₀₂(x) + f₃₀₃(x) + f₃₀₄(x)
    # D = 4, R = 0
    c₄₀ₙ = rand(SVector{5,S})
    function f₄₀ₙ(x, n)
        local y = SVector{5,S}(1, x...)
        return c₄₀ₙ[n] * y[n] * Form{4,0,S}((1,))
    end
    f₄₀(x) = sum(f₄₀ₙ(x, n) for n in 1:5)
    # D = 5, R = 0
    c₅₀ₙ = rand(SVector{6,S})
    function f₅₀ₙ(x, n)
        local y = SVector{6,S}(1, x...)
        return c₅₀ₙ[n] * y[n] * Form{5,0,S}((1,))
    end
    f₅₀(x) = sum(f₅₀ₙ(x, n) for n in 1:5)

    function f(x)
        D == 0 && R == 0 && return f₀₀(x)
        D == 1 && R == 0 && return f₁₀(x)
        D == 1 && R == 1 && return f₁₁(x)
        D == 2 && R == 0 && return f₂₀(x)
        D == 2 && R == 1 && return f₂₁(x)
        D == 2 && R == 2 && return f₂₂(x)
        D == 3 && R == 0 && return f₃₀(x)
        D == 4 && R == 0 && return f₄₀(x)
        D == 5 && R == 0 && return f₅₀(x)
        @assert false
    end
    T = S
    f(zero(SVector{D,S}))::Form{D,R,T}
    f̃ = sample(Fun{D,P,R,D,S,T}, f, mfd)

    xs = [
        collect(get_coords(mfd, R))
        [random_point(Val(R), mfd) for i in 1:10]
    ]
    for x in xs
        fx = f(x)::Form{D,R,T}
        f̃x = evaluate(f̃, x)::Form{D,R,T}
        Ex = norm(f̃x - fx)
        @test 1 + Ex ≈ 1
    end

    #TODO if R < D
    #TODO     function df(x)
    #TODO         @assert m !== missing
    #TODO         local r = m
    #TODO         return r::Form{D,R + 1,T}
    #TODO     end
    #TODO 
    #TODO     df̃ = sample(Fun{D,P,R + 1,D,S,T}, df, mfd)
    #TODO     df̃::Fun{D,P,R + 1,D,S,T}
    #TODO 
    #TODO     d̃f̃ = deriv(f̃₁)
    #TODO     d̃f̃::Fun{D,P,R + 1,D,S,T}
    #TODO 
    #TODO     @test norm(d̃f̃ - df̃, Inf) ≤ sqrt(eps(T))
    #TODO 
    #TODO     # We cannot evaluate derivatives pointwise since there are not
    #TODO     # enough degrees of freedom to describe all form components in
    #TODO     # all directions. The discrete function has discontinuous
    #TODO     # normals across interfaces.
    #TODO end
    #TODO 
    #TODO if R > 0
    #TODO     function δf(x)
    #TODO         @assert m′ !== missing
    #TODO         local r = bitsign(R) * bitsign(R * (D - R)) * m′
    #TODO         return r::Form{D,R - 1,T}
    #TODO     end
    #TODO 
    #TODO     δf̃ = sample(Fun{D,P,R - 1,D,S,T}, δf, mfd)
    #TODO     δf̃::Fun{D,P,R - 1,D,S,T}
    #TODO 
    #TODO     δ̃f̃ = coderiv(f̃)
    #TODO     δ̃f̃::Fun{D,P,R - 1,D,S,T}
    #TODO 
    #TODO     if !(norm(δ̃f̃ - δf̃, Inf) ≤ sqrt(eps(T)))
    #TODO         @show mfd
    #TODO         @show D P R
    #TODO         @show get_coords(mfd, 0) get_coords(mfd, R)
    #TODO         @show f̃ δ̃f̃ δf̃
    #TODO     end
    #TODO     @test norm(δ̃f̃ - δf̃, Inf) ≤ sqrt(eps(T))
    #TODO end
end

@testset "Project a function D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P
    P == Pr || continue

    S = Float64
    mfd = simplex_manifold(Val(D), S)
    # for level in 1:2
    #     mfd = refined_manifold(mfd; optimize_mesh=false)
    # end
    # mfd = large_hypercube_manifold(Val(D), S; nelts=ntuple(d -> 2, D),
    #                                optimize_mesh=false)

    # m = R + 1 > D ? missing : rand(Form{D,R + 1,S})
    # m′ = R - 1 < 0 ? missing : rand(Form{D,R - 1,S})
    # b = rand(Form{D,R,S})
    # function f(x)
    #     r = b
    #     if m !== missing
    #         r += m ⋅ Form{D,1,S}(x)
    #     end
    #     if m′ !== missing
    #         r += ⋆(⋆m′ ⋅ Form{D,1,S}(x))
    #     end
    #     return r::Form{D,R,T}
    # end

    # nomenclature: fDRN where D=dim, R=rank, N=basis element
    # D = 0, R = 0
    c₀₀₁ = rand(S)
    f₀₀₁(x) = c₀₀₁ * Form{0,0,S}((1,)) # 1
    f₀₀(x) = f₀₀₁(x)
    # D = 1, R = 0
    c₁₀₁ = rand(S)
    f₁₀₁(x) = c₁₀₁ * Form{1,0,S}((1,)) # 1
    c₁₀₂ = rand(S)
    f₁₀₂(x) = c₁₀₂ * x[1] * Form{1,0,S}((1,)) # x
    f₁₀(x) = f₁₀₁(x) + f₁₀₂(x)
    # D = 1, R = 1
    c₁₁₁ = rand(S)
    f₁₁₁(x) = c₁₁₁ * Form{1,1,S}((1,)) # dx
    f₁₁(x) = f₁₁₁(x)
    # D = 2, R = 0
    c₂₀₁ = rand(S)
    f₂₀₁(x) = c₂₀₁ * Form{2,0,S}((1,)) # 1
    c₂₀₂ = rand(S)
    f₂₀₂(x) = c₂₀₂ * x[1] * Form{2,0,S}((1,)) # x
    c₂₀₃ = rand(S)
    f₂₀₃(x) = c₂₀₃ * x[2] * Form{2,0,S}((1,)) # y
    f₂₀(x) = f₂₀₁(x) + f₂₀₂(x) + f₂₀₃(x)
    # D = 2, R = 1
    c₂₁₁ = S(1) # rand(S)
    f₂₁₁(x) = c₂₁₁ * Form{2,1,S}((1, 0)) # dx
    c₂₁₂ = S(0) # rand(S)
    f₂₁₂(x) = c₂₁₂ * Form{2,1,S}((0, 1)) # dy
    c₂₁₃ = S(0) # rand(S)
    f₂₁₃(x) = c₂₁₃ * (-x[2] * Form{2,1,S}((1, 0)) + x[1] * Form{2,1,S}((0, 1))) # - y dx + x dy
    f₂₁(x) = f₂₁₁(x) + f₂₁₂(x) + f₂₁₃(x)
    # D = 2, R = 2
    c₂₂₁ = rand(S)
    f₂₂₁(x) = c₂₂₁ * Form{2,2,S}((1, 1)) # dx ∧ dy
    f₂₂(x) = f₂₂₁(x)

    function f(x)
        D == 0 && R == 0 && return f₀₀(x)
        D == 1 && R == 0 && return f₁₀(x)
        D == 1 && R == 1 && return f₁₁(x)
        D == 2 && R == 0 && return f₂₀(x)
        D == 2 && R == 1 && return f₂₁(x)
        D == 2 && R == 2 && return f₂₂(x)
        D == 3 && R == 0 && return f₃₀(x)
        D == 4 && R == 0 && return f₄₀(x)
        D == 5 && R == 0 && return f₅₀(x)
        @assert false
    end
    T = S
    f(zero(SVector{D,S}))::Form{D,R,T}
    f̃ = project(Fun{D,P,R,D,S,T}, f, mfd)

    xs = [
        collect(get_coords(mfd, R))
        [random_point(Val(R), mfd) for i in 1:10]
    ]
    for x in xs
        fx = f(x)::Form{D,R,T}
        f̃x = evaluate(f̃, x)::Form{D,R,T}
        Ex = norm(f̃x - fx)
        if !(1 + Ex ≈ 1)
            @show D R get_coords(mfd, R) f̃ x fx f̃x Ex
        end
        @test 1 + Ex ≈ 1
    end

    # TODO: check derivatives etc; translate code from sampling
end
