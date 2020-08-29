using DDF

using ComputedFieldTypes
using DifferentialForms
using LinearAlgebra
using StaticArrays

@testset "Evaluate a function D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P, all R
    (P == Pr && R == 0) || continue

    S = Float64
    mfd = simplex_manifold(Val(D), S)

    T = S

    f0 = zero(Fun{D,P,R,S,T}, mfd)
    nvalues = nsimplices(mfd, P == Pr ? R : D - R)
    f1 = Fun{D,P,R,S,T}(mfd, [T(i) for i in 1:nvalues])
    f2 = Fun{D,P,R,S,T}(mfd, rand(T, nvalues))
    a = rand(T)

    for i in 1:nsimplices(mfd, P == Pr ? R : D - R)
        x = SVector{D,S}(@view mfd.coords[i, :])
        @test evaluate(f0, x) == zero(Form{D,R,T})
        @test evaluate(f1 + f2, x) ≈ evaluate(f1, x) + evaluate(f2, x)
        @test evaluate(a * f1, x) ≈ a * evaluate(f1, x)
    end

    for i in 1:10
        x = random_point(Val(R), mfd)
        @test evaluate(f0, x) == zero(Form{D,R,T})
        @test evaluate(f1 + f2, x) ≈ evaluate(f1, x) + evaluate(f2, x)
        @test evaluate(a * f1, x) ≈ a * evaluate(f1, x)
    end
end

@testset "Sample a function D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P, all R
    (P == Pr && R == 0) || continue

    S = Float64
    mfd = simplex_manifold(Val(D), S)

    # f(x) = Form{D,R,S}((prod(sin(2π * x[d]) for d in 1:D),))
    b = rand(Form{D,0,S})
    if D == 0
        # f(x) = b
        f = x -> b
    else
        m = rand(Form{D,1,S})
        # f(x) = m ⋅ Form{D,1,S}(x) + b
        f = x -> m ⋅ Form{D,1,S}(x) + b
    end
    T = S
    f(zero(SVector{D,S}))::Form{D,R,T}
    f̃ = sample(Fun{D,P,R,S,T}, f, mfd)

    for i in 1:nsimplices(mfd, R)
        x = SVector{D,S}(@view mfd.coords[i, :])
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

@testset "Project a function D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all P, all R
    (P == Pr && R == 0) || continue

    S = Float64
    mfd = simplex_manifold(Val(D), S)

    # f(x) = Form{D,R,S}((prod(sin(2π * x[d]) for d in 1:D),))
    b = rand(Form{D,0,S})
    if D == 0
        f = x -> b
    else
        m = rand(Form{D,1,S})
        f = x -> m ⋅ Form{D,1,S}(x) + b
    end
    T = S
    f(zero(SVector{D,S}))::Form{D,R,T}
    f̃ = project(Fun{D,P,R,S,T}, f, mfd)

    for i in 1:nsimplices(mfd, R)
        x = SVector{D,S}(@view mfd.coords[i, :])
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
