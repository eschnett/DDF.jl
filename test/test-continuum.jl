using DDF

using Bernstein
using ComputedFieldTypes
using DifferentialForms
using LinearAlgebra
using StaticArrays

@testset "Basis functions D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P
    P == Pr || continue

    S = Float64
    mfd = simplex_manifold(Val(D), S)

    xs = SVector{D + 1,SVector{D,S}}(mfd.coords[0])
    x2λ = cartesian2barycentric_setup(xs)
    dλ2dx = Continuum.dbarycentric2dcartesian_setup(Form{D,R}, xs)

    for n in 1:nsimplices(mfd, P == Pr ? R : D - R)
        for i in 1:nsimplices(mfd, 0)
            x = mfd.coords[0][i]
            bx = Continuum.basis_x(Form{D,R}, x2λ, dλ2dx, n, x)
        end

        for i in 1:nsimplices(mfd, 0)
            x = mfd.coords[0][i]
            bx = Continuum.basis_x(Form{D,R}, x2λ, dλ2dx, n, x)
            if R == 0
                @test bx ≈ Form{D,R}((i == n ? mfd.volumes[R][n] : 0,))
            elseif R == D
                @test bx ≈ Form{D,R}((mfd.volumes[R][n],))
            elseif D == 1 && R == 1
                @test bx ≈ Form{D,R}((mfd.volumes[R][n],))
            elseif D == 2 && R == 1
                v1, v2 = [(1, 2), (1, 3), (2, 3)][n]
                dx = mfd.coords[0][v2] - mfd.coords[0][v1]
                @test bx ≈ Form{D,R}((i ∈ (v1, v2)) * dx)
            elseif D == 3 && R == 1
                v1, v2 = [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4)][n]
                dx = mfd.coords[0][v2] - mfd.coords[0][v1]
                @test bx ≈ Form{D,R}((i ∈ (v1, v2)) * dx)
            elseif D == 4 && R == 1
                v1, v2 = [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4),
                          (1, 5), (2, 5), (3, 5), (4, 5)][n]
                dx = mfd.coords[0][v2] - mfd.coords[0][v1]
                @test bx ≈ Form{D,R}((i ∈ (v1, v2)) * dx)
            elseif D == 5 && R == 1
                v1, v2 = [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4),
                          (1, 5), (2, 5), (3, 5), (4, 5), (1, 6), (2, 6),
                          (3, 6), (4, 6), (5, 6)][n]
                dx = Form{D,R}(mfd.coords[0][v2] - mfd.coords[0][v1])
                @test bx ≈ (i ∈ (v1, v2)) * dx
            elseif D == 3 && R == 2
                v1, v2, v3 = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)][n]
                dx = Form{D,1}(mfd.coords[0][v2] - mfd.coords[0][v1]) ∧
                     Form{D,1}(mfd.coords[0][v3] - mfd.coords[0][v1]) / 2
                @test bx ≈ (i ∈ (v1, v2, v3)) * dx
            elseif D == 4 && R == 2
                v1, v2, v3 = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
                              (1, 2, 5), (1, 3, 5), (2, 3, 5), (1, 4, 5),
                              (2, 4, 5), (3, 4, 5)][n]
                dx = Form{D,1}(mfd.coords[0][v2] - mfd.coords[0][v1]) ∧
                     Form{D,1}(mfd.coords[0][v3] - mfd.coords[0][v1]) / 2
                @test bx ≈ (i ∈ (v1, v2, v3)) * dx
            elseif D == 5 && R == 2
                v1, v2, v3 = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
                              (1, 2, 5), (1, 3, 5), (2, 3, 5), (1, 4, 5),
                              (2, 4, 5), (3, 4, 5), (1, 2, 6), (1, 3, 6),
                              (2, 3, 6), (1, 4, 6), (2, 4, 6), (3, 4, 6),
                              (1, 5, 6), (2, 5, 6), (3, 5, 6), (4, 5, 6)][n]
                dx = Form{D,1}(mfd.coords[0][v2] - mfd.coords[0][v1]) ∧
                     Form{D,1}(mfd.coords[0][v3] - mfd.coords[0][v1]) / 2
                @test bx ≈ (i ∈ (v1, v2, v3)) * dx
            elseif D == 4 && R == 3
                v1, v2, v3, v4 = [(1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 4, 5),
                                  (1, 3, 4, 5), (2, 3, 4, 5)][n]
                dx = Form{D,1}(mfd.coords[0][v2] - mfd.coords[0][v1]) ∧
                     Form{D,1}(mfd.coords[0][v3] - mfd.coords[0][v1]) ∧
                     Form{D,1}(mfd.coords[0][v4] - mfd.coords[0][v1]) / 6
                @test bx ≈ (i ∈ (v1, v2, v3, v4)) * dx
            end
        end

        #TODO for i in 1:nsimplices(mfd, P == Pr ? R : D - R)
        #TODO     x = mfd.coords[R][i]
        #TODO     bx = Continuum.basis_x(Form{D,R}, x2λ, dλ2dx, n, x)
        #TODO     @test bx ≈ Form{D,R}((n == i,))
        #TODO end
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
    fi = Fun{D,P,R,D,S,T}(mfd, [T(i) for i in 1:nvalues])
    fx = Fun{D,P,R,D,S,SVector{D,T}}(mfd, mfd.coords[R])
    f1 = Fun{D,P,R,D,S,T}(mfd, rand(T, nvalues))
    f2 = Fun{D,P,R,D,S,T}(mfd, rand(T, nvalues))
    a = rand(T)

    for i in 1:nsimplices(mfd, P == Pr ? R : D - R)
        x = mfd.coords[R][i]
        if R == 0
            @test evaluate(fi, x) ≈ Form{D,R}((T(i),))
            @test evaluate(fx, x) ≈ Form{D,R}((x,))
        end
        @test evaluate(f0, x) == zero(Form{D,R,T})
        @test evaluate(f1 + f2, x) ≈ evaluate(f1, x) + evaluate(f2, x)
        @test evaluate(a * f1, x) ≈ a * evaluate(f1, x)
    end

    for i in 1:10
        x = random_point(Val(D), mfd)
        if R == 0
            @test evaluate(fx, x) ≈ Form{D,R}((x,))
        end
        @test evaluate(f0, x) == zero(Form{D,R,T})
        @test evaluate(f1 + f2, x) ≈ evaluate(f1, x) + evaluate(f2, x)
        @test evaluate(a * f1, x) ≈ a * evaluate(f1, x)
    end
end

@testset "Sample a function D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P
    P == Pr || continue

    S = Float64
    mfd = simplex_manifold(Val(D), S)

    # f(x) = Form{D,R,D,S}((prod(sin(2π * x[d]) for d in 1:D),))
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
        return b::Form{D,R,T}
    end
    T = S
    f(zero(SVector{D,S}))::Form{D,R,T}
    f̃ = sample(Fun{D,P,R,D,S,T}, f, mfd)

    for i in 1:nsimplices(mfd, R)
        x = mfd.coords[R][i]
        fx = f(x)::Form{D,R,T}
        f̃x = evaluate(f̃, x)::Form{D,R,T}
        Ex = norm(f̃x - fx)
        if R == 0
            @test 1 + Ex ≈ 1
        else
            # Sampling is only accurate for 0-forms
            @test abs(Ex) ≤ 2
        end
    end

    for i in 1:10
        x = random_point(Val(R), mfd)
        fx = f(x)::Form{D,R,T}
        f̃x = evaluate(f̃, x)::Form{D,R,T}
        Ex = norm(f̃x - fx)
        if R == 0
            @test 1 + Ex ≈ 1
        else
            # Sampling is only accurate for 0-forms
            @test abs(Ex) ≤ 2
        end
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

    # f(x) = Form{D,R,D,S}((prod(sin(2π * x[d]) for d in 1:D),))
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

    for i in 1:nsimplices(mfd, R)
        x = mfd.coords[R][i]
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
