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

    basis_λ = Continuum.basis_λ

    if D == 0
        # vertices: 1
        λ1 = SVector{N,S}(1)
        if R == 0
            @test basis_λ(Form{D,R}, 1, λ1) == Form{D + 1,0}((1,))
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
        λ1 = SVector{N,S}(1, 0)
        λ2 = SVector{N,S}(0, 1)
        if R == 0
            @test basis_λ(Form{D,R}, 1, λ1) == Form{D + 1,R}((1,))
            @test basis_λ(Form{D,R}, 1, λ2) == Form{D + 1,R}((0,))
            @test basis_λ(Form{D,R}, 2, λ1) == Form{D + 1,R}((0,))
            @test basis_λ(Form{D,R}, 2, λ2) == Form{D + 1,R}((1,))
            @test basis_λ(Form{D,R}, 1, (λ1 + λ2) / 2) ==
                  Form{D + 1,R}((1,)) / 2
            @test basis_λ(Form{D,R}, 2, (λ1 + λ2) / 2) ==
                  Form{D + 1,R}((1,)) / 2
        elseif R == 1
            @test d.values == [-1 1]
            @test basis_λ(Form{D,R}, 1, λ1) == Form{D + 1,R}((-1, 1))
            @test basis_λ(Form{D,R}, 1, λ2) == Form{D + 1,R}((-1, 1))
            @test basis_λ(Form{D,R}, 1, (λ1 + λ2) / 2) == Form{D + 1,R}((-1, 1))
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
        λ1 = SVector{N,S}(1, 0, 0)
        λ2 = SVector{N,S}(0, 1, 0)
        λ3 = SVector{N,S}(0, 0, 1)
        if R == 0
            @test basis_λ(Form{D,R}, 1, λ1) == Form{D + 1,R}((1,))
            @test basis_λ(Form{D,R}, 1, λ2) == Form{D + 1,R}((0,))
            @test basis_λ(Form{D,R}, 1, λ3) == Form{D + 1,R}((0,))
            @test basis_λ(Form{D,R}, 2, λ1) == Form{D + 1,R}((0,))
            @test basis_λ(Form{D,R}, 2, λ2) == Form{D + 1,R}((1,))
            @test basis_λ(Form{D,R}, 2, λ3) == Form{D + 1,R}((0,))
            @test basis_λ(Form{D,R}, 3, λ1) == Form{D + 1,R}((0,))
            @test basis_λ(Form{D,R}, 3, λ2) == Form{D + 1,R}((0,))
            @test basis_λ(Form{D,R}, 3, λ3) == Form{D + 1,R}((1,))
            @test basis_λ(Form{D,R}, 1, (λ1 + λ2) / 2) ==
                  Form{D + 1,R}((1,)) / 2
            @test basis_λ(Form{D,R}, 1, (λ1 + λ3) / 2) ==
                  Form{D + 1,R}((1,)) / 2
            @test basis_λ(Form{D,R}, 1, (λ2 + λ3) / 2) ==
                  Form{D + 1,R}((0,)) / 2
            @test basis_λ(Form{D,R}, 1, (λ1 + λ2 + λ3) / 3) ==
                  Form{D + 1,R}((1,)) / 3
        elseif R == 1
            @test d.values == [
                -1 1 0
                -1 0 1
                0 -1 1
            ]
            @test basis_λ(Form{D,R}, 1, λ1) == Form{D + 1,R}((-1, 1, 0))
            @test basis_λ(Form{D,R}, 1, λ2) == Form{D + 1,R}((-1, 1, 0))
            @test basis_λ(Form{D,R}, 1, λ3) == Form{D + 1,R}((0, 0, 0))
            @test basis_λ(Form{D,R}, 2, λ1) == Form{D + 1,R}((-1, 0, 1))
            @test basis_λ(Form{D,R}, 2, λ2) == Form{D + 1,R}((0, 0, 0))
            @test basis_λ(Form{D,R}, 2, λ3) == Form{D + 1,R}((-1, 0, 1))
            @test basis_λ(Form{D,R}, 3, λ1) == Form{D + 1,R}((0, 0, 0))
            @test basis_λ(Form{D,R}, 3, λ2) == Form{D + 1,R}((0, -1, 1))
            @test basis_λ(Form{D,R}, 3, λ3) == Form{D + 1,R}((0, -1, 1))
            @test basis_λ(Form{D,R}, 1, (λ1 + λ2) / 2) ==
                  Form{D + 1,R}((-1, 1, 0))
            @test basis_λ(Form{D,R}, 1, (λ1 + λ3) / 2) ==
                  Form{D + 1,R}((-1, 1, 0)) / 2
            @test basis_λ(Form{D,R}, 1, (λ2 + λ3) / 2) ==
                  Form{D + 1,R}((-1, 1, 0)) / 2
            @test basis_λ(Form{D,R}, 1, (λ1 + λ2 + λ3) / 3) ==
                  Form{D + 1,R}((-1, 1, 0)) * 2 / 3
        elseif R == 2
            @test d.values == [1 -1 1]
            # TODO: These `... / S(2)` seem wrong
            @test basis_λ(Form{D,R}, 1, λ1) == Form{D + 1,R}((1, -1, 1)) / S(2)
            @test basis_λ(Form{D,R}, 1, λ2) == Form{D + 1,R}((1, -1, 1)) / S(2)
            @test basis_λ(Form{D,R}, 1, λ3) == Form{D + 1,R}((1, -1, 1)) / S(2)
        end
    end
end

@testset "Basis functions D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P
    P == Pr || continue

    S = Float64
    mfd = simplex_manifold(Val(D), S)

    xs = SVector{D + 1,SVector{D,S}}(get_coords(mfd)[i]
                                     for i in ID{0}(1):ID{0}(D + 1))
    x2λ = cartesian2barycentric_setup(xs)
    dλ2dx = Continuum.dbarycentric2dcartesian_setup(Form{D,R}, xs)

    VR = IDVector{R}
    I0(is...) = map(ID{0}, is)

    for n in axes(get_simplices(mfd, P == Pr ? R : D - R), 2)
        for i in axes(get_simplices(mfd, 0), 1)
            x = get_coords(mfd)[i]
            bx = Continuum.basis_x(Form{D,R}, x2λ, dλ2dx, Int(n), x)
            if R == 0
                @test bx ≈
                      Form{D,R}((Int(i) == Int(n) ? get_volumes(mfd, R)[n] : 0,))
            elseif R == D
                @test bx ≈ Form{D,R}((get_volumes(mfd, R)[n],))
            elseif D == 1 && R == 1
                @test bx ≈ Form{D,R}((get_volumes(mfd, R)[n],))
            elseif D == 2 && R == 1
                v1, v2 = VR([I0(1, 2), I0(1, 3), I0(2, 3)])[n]
                dx = get_coords(mfd)[v2] - get_coords(mfd)[v1]
                @test bx ≈ Form{D,R}((i in (v1, v2)) * dx)
            elseif D == 3 && R == 1
                v1, v2 = VR([I0(1, 2), I0(1, 3), I0(2, 3), I0(1, 4), I0(2, 4),
                             I0(3, 4)])[n]
                dx = get_coords(mfd)[v2] - get_coords(mfd)[v1]
                @test bx ≈ Form{D,R}((i in (v1, v2)) * dx)
            elseif D == 4 && R == 1
                v1, v2 = VR([I0(1, 2), I0(1, 3), I0(2, 3), I0(1, 4), I0(2, 4),
                             I0(3, 4), I0(1, 5), I0(2, 5), I0(3, 5), I0(4, 5)])[n]
                dx = get_coords(mfd)[v2] - get_coords(mfd)[v1]
                @test bx ≈ Form{D,R}((i in (v1, v2)) * dx)
            elseif D == 5 && R == 1
                v1, v2 = VR([I0(1, 2), I0(1, 3), I0(2, 3), I0(1, 4), I0(2, 4),
                             I0(3, 4), I0(1, 5), I0(2, 5), I0(3, 5), I0(4, 5),
                             I0(1, 6), I0(2, 6), I0(3, 6), I0(4, 6), I0(5, 6)])[n]
                dx = Form{D,R}(get_coords(mfd)[v2] - get_coords(mfd)[v1])
                @test bx ≈ (i in (v1, v2)) * dx
            elseif D == 3 && R == 2
                v1, v2, v3 = VR([I0(1, 2, 3), I0(1, 2, 4), I0(1, 3, 4),
                                 I0(2, 3, 4)])[n]
                dx = Form{D,1}(get_coords(mfd)[v2] - get_coords(mfd)[v1]) ∧
                     Form{D,1}(get_coords(mfd)[v3] - get_coords(mfd)[v1]) / 2
                @test bx ≈ (i in (v1, v2, v3)) * dx
            elseif D == 4 && R == 2
                v1, v2, v3 = VR([I0(1, 2, 3), I0(1, 2, 4), I0(1, 3, 4),
                                 I0(2, 3, 4), I0(1, 2, 5), I0(1, 3, 5),
                                 I0(2, 3, 5), I0(1, 4, 5), I0(2, 4, 5),
                                 I0(3, 4, 5)])[n]
                dx = Form{D,1}(get_coords(mfd)[v2] - get_coords(mfd)[v1]) ∧
                     Form{D,1}(get_coords(mfd)[v3] - get_coords(mfd)[v1]) / 2
                @test bx ≈ (i in (v1, v2, v3)) * dx
            elseif D == 5 && R == 2
                v1, v2, v3 = VR([I0(1, 2, 3), I0(1, 2, 4), I0(1, 3, 4),
                                 I0(2, 3, 4), I0(1, 2, 5), I0(1, 3, 5),
                                 I0(2, 3, 5), I0(1, 4, 5), I0(2, 4, 5),
                                 I0(3, 4, 5), I0(1, 2, 6), I0(1, 3, 6),
                                 I0(2, 3, 6), I0(1, 4, 6), I0(2, 4, 6),
                                 I0(3, 4, 6), I0(1, 5, 6), I0(2, 5, 6),
                                 I0(3, 5, 6), I0(4, 5, 6)])[n]
                dx = Form{D,1}(get_coords(mfd)[v2] - get_coords(mfd)[v1]) ∧
                     Form{D,1}(get_coords(mfd)[v3] - get_coords(mfd)[v1]) / 2
                @test bx ≈ (i in (v1, v2, v3)) * dx
            elseif D == 4 && R == 3
                v1, v2, v3, v4 = VR([I0(1, 2, 3, 4), I0(1, 2, 3, 5),
                                     I0(1, 2, 4, 5), I0(1, 3, 4, 5),
                                     I0(2, 3, 4, 5)])[n]
                dx = Form{D,1}(get_coords(mfd)[v2] - get_coords(mfd)[v1]) ∧
                     Form{D,1}(get_coords(mfd)[v3] - get_coords(mfd)[v1]) ∧
                     Form{D,1}(get_coords(mfd)[v4] - get_coords(mfd)[v1]) / 6
                @test bx ≈ (i in (v1, v2, v3, v4)) * dx
            end
        end
    end
end

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

    if R < D
        dfi = deriv(fi)
        if D == 1
            # vertices: 1 2
            # edges:    (1,2)
            x1 = get_coords(mfd)[ID{0}(1)]
            x2 = get_coords(mfd)[ID{0}(2)]
            fi1 = evaluate(fi, x1)
            fi2 = evaluate(fi, x2)
            dfi1 = evaluate(dfi, x1)
            dfi2 = evaluate(dfi, x2)
            if R == 0
                @test dfi1 ≈ (-fi1 + fi2)[] * Form{D,1}(-x1 + x2)
                @test dfi2 ≈ (-fi1 + fi2)[] * Form{D,1}(-x1 + x2)
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
                @test dfi1 ≈
                      (-fi1 + fi2)[] * Form{D,1}(-x1 + x2) +
                      (-fi1 + fi3)[] * Form{D,1}(-x1 + x3)
                @test dfi2 ≈
                      (-fi1 + fi2)[] * Form{D,1}(-x1 + x2) +
                      (-fi2 + fi3)[] * Form{D,1}(-x2 + x3)
                @test dfi3 ≈
                      (fi1 - fi3)[] * Form{D,1}(x1 - x3) +
                      (-fi2 + fi3)[] * Form{D,1}(-x2 + x3)
            elseif R == 1
                #TODO 
                #TODO STARTING POINT IS 1-FORMS
                #TODO 
                #TODO @test dfi1 ≈
                #TODO       (fi1 - fi2 + fi3)[] * Form{D,1}(-x1 + x2) ∧
                #TODO       Form{D,1}(-x1 + x3)
                #TODO @test dfi2 ≈
                #TODO       (fi3 - fi2 + fi3)[] * Form{D,1}(-x1 + x2) ∧
                #TODO       Form{D,1}(-x1 + x3)
                #TODO @test dfi1 ≈
                #TODO       (fi1 - fi2 + fi3)[] * Form{D,1}(-x1 + x2) ∧
                #TODO       Form{D,1}(-x1 + x3)
            end
        end
    end
end

@testset "Sample a function D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P
    P == Pr || continue

    S = Float64
    mfd = simplex_manifold(Val(D), S)
    for level in 1:1
        mfd = refined_manifold(mfd; optimize_mesh=false)
    end

    # f(x) = Form{D,R,D,S}((prod(sin(2π * x[d]) for d  in  1:D),))
    m = R + 1 > D ? missing : rand(Form{D,R + 1,S})
    m′ = R - 1 < 0 ? missing : rand(Form{D,R - 1,S})
    b = rand(Form{D,R,S})
    function f(x)
        local r = b
        if m !== missing
            r += m ⋅ Form{D,1,S}(x)
        end
        if m′ !== missing
            r += m′ ∧ Form{D,1,S}(x)
        end
        return r::Form{D,R,T}
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
        if R == 0
            @test 1 + Ex ≈ 1
        else
            # Sampling is only accurate for 0-forms
            @test abs(Ex) ≤ 3
        end
    end

    if R + 1 ≤ D
        function df(x)
            @assert m !== missing
            local r = m
            return r::Form{D,R + 1,T}
        end

        df̃ = sample(Fun{D,P,R + 1,D,S,T}, df, mfd)
        df̃::Fun{D,P,R + 1,D,S,T}

        d̃f̃ = deriv(f̃)
        d̃f̃::Fun{D,P,R + 1,D,S,T}

        @test norm(d̃f̃ - df̃, Inf) ≤ sqrt(eps(T))

        # We cannot evaluate derivatives pointwise since there are not
        # enough degrees of freedom to describe all form components in
        # all directions. The discrete function has discontinuous
        # normals across interfaces.
    end

    if R > 0
        δf̃ = coderiv(f̃)
        δf̃::Fun{D,P,R - 1,D,S,T}
        # TODO: test this!
    end
end

@testset "Project a function D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: Projecting only works for R==0 and R==D
    (R == 0 || R == D) || continue

    # TODO: all P
    P == Pr || continue

    S = Float64
    mfd = simplex_manifold(Val(D), S)

    # f(x) = Form{D,R,D,S}((prod(sin(2π * x[d]) for d  in  1:D),))
    m = R + 1 > D ? missing : rand(Form{D,R + 1,S})
    m′ = R - 1 < 0 ? missing : rand(Form{D,R - 1,S})
    b = rand(Form{D,R,S})
    function f(x)
        r = b
        if m !== missing
            r += m ⋅ Form{D,1,S}(x)
        end
        if m′ !== missing
            r += m′ ∧ Form{D,1,S}(x)
        end
        # N = length(Form{D,R})
        # b = Form{D,R,T}(ntuple(n -> n == 1, N))
        return b::Form{D,R,T}
    end
    T = S
    f(zero(SVector{D,S}))::Form{D,R,T}
    f̃ = project(Fun{D,P,R,D,S,T}, f, mfd)

    # b1 = (1-x) dy
    # b2 = (1-y) dx
    # b3 = (x+y) (dy-dx)

    # b2 - b3 = (1 + x) dx - (x+y) dy

    for i in axes(get_simplices(mfd, R), 2)
        x = get_coords(mfd, R)[i]
        fx = f(x)::Form{D,R,T}
        f̃x = evaluate(f̃, x)::Form{D,R,T}
        Ex = norm(f̃x - fx)
        @test 1 + Ex ≈ 1
    end

    for i in 1:10
        x = random_point(Val(R), mfd)
        fx = f(x)::Form{D,R,T}
        f̃x = evaluate(f̃, x)
        f̃x::Form{D,R,T}
        Ex = norm(f̃x - fx)
        @test 1 + Ex ≈ 1
    end
end
