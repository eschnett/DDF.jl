using DDF

using ComputedFieldTypes
using DifferentialForms
using LinearAlgebra
using StaticArrays

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
            # Sampling is only accurate on vertices
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
            # Sampling is only accurate on vertices
            @test abs(Ex) ≤ 2
        end
    end
end

@testset "Project a function D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P, all R
    (P == Pr && R == 0) || continue

    S = Float64
    mfd = simplex_manifold(Val(D), S)

    # f(x) = Form{D,R,D,S}((prod(sin(2π * x[d]) for d in 1:D),))
    b = rand(Form{D,0,S})
    if D == 0
        f = x -> b
    else
        m = rand(Form{D,1,S})
        f = x -> m ⋅ Form{D,1,S}(x) + b
    end
    T = S
    f(zero(SVector{D,S}))::Form{D,R,T}
    f̃ = project(Fun{D,P,R,D,S,T}, f, mfd)

    for i in 1:nsimplices(mfd, R)
        x = mfd.coords[0][i]
        fx = f(x)::Form{D,R,T}
        f̃x = evaluate(f̃, x)
        f̃x::Form{D,R,T}
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
