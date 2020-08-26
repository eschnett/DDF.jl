using DDF

using LinearAlgebra
using StaticArrays
using Test

@testset "Manifold ops D=$D P=$P R=$R" for D in 0:Dmax, P in (Pr, Dl), R in 0:D
    # S = Rational{Int128}
    S = Float64

    mfds = [empty_manifold(Val(D), S), simplex_manifold(Val(D), S),
            hypercube_manifold(Val(D), S),
            delaunay_hypercube_manifold(Val(D), S)]

    for mfd in mfds
        funs = Fun{D,P,R,S}[]
        push!(funs, zero(Fun{D,P,R,S,S}, mfd))
        e = id(Fun{D,P,0,S,SVector{D,S}}, mfd)
        if R == 0
            push!(funs, e)
        end

        s = P == Pr ? +1 : -1

        # Boundary

        R2 = R - 2s
        if 0 <= R2 <= D
            b1 = boundary(Val(P), Val(R), mfd)
            b2 = boundary(Val(P), Val(R - s), mfd)
            b21 = b2 * b1
            b21::Op{D,P,R2,P,R,Int8}
            @test iszero(b21)
        end

        R1 = R - s
        if 0 <= R1 <= D
            b = boundary(Val(P), Val(R), mfd)
            for f in funs
                bf = b * f
                T = eltype(f)
                bf::Fun{D,P,R1,S,T}
            end
        end

        # Derivative
        R2 = R + 2s
        if 0 <= R2 <= D
            d1 = deriv(Val(P), Val(R), mfd)
            d2 = deriv(Val(P), Val(R + s), mfd)
            d21 = d2 * d1
            d21::Op{D,P,R2,P,R,Int8}
            @test iszero(d21)
        end

        R1 = R + s
        if 0 <= R1 <= D
            d = deriv(Val(P), Val(R), mfd)
            for f in funs
                df = d * f
                T = eltype(f)
                df::Fun{D,P,R1,S,T}
            end
        end

        # Only if completely well-centred [arXiv:0802.2108 [cs.CG]]
        if mfd.name ∉ ["hypercube manifold", "delaunay hypercube manifold"]

            # Hodge
            h = hodge(Val(P), Val(R), mfd)
            h′ = hodge(Val(!P), Val(R), mfd)
            @test invhodge(Val(P), Val(R), mfd) == h′
            @test invhodge(Val(!P), Val(R), mfd) == h
            h21 = h′ * h
            h21::Op{D,P,R,P,R,S}
            if !isempty(h21)
                @test norm((h21 - one(h21)).values, Inf) <= 10eps(S)
            end

        end

    end
end
