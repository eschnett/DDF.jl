using DDF

using ComputedFieldTypes
using DifferentialForms
using StaticArrays

@testset "Sample a function D=$D P=$P R=$R" for D in 0:Dmax,
P in (Pr, Dl),
R in 0:D

    # TODO: all R, all P
    (R == 0 && P == Pr) || continue

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
end

@warn "CONTINUE HERE"

@testset "evaluate" begin end
