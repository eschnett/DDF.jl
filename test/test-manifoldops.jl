using DDF

using StaticArrays
using Test

@testset "Manifold ops D=$D P=$P R=$R" for D in 0:Dmax, P in (Pr, Dl), R in 0:D
    T = Rational{Int128}

    mfds = [empty_manifold(Val(D), T), simplex_manifold(Val(D), T),
            hypercube_manifold(Val(D), T)]

    for mfd in mfds
        funs = Fun{D,P,R,T,T}[]
        push!(funs, zero(Fun{D,P,R,T,T}, mfd))
        e = id(Fun{D,P,0,T,SVector{D,T}}, mfd)
        if R == 0
            for d in 1:D
                push!(funs, map(x -> x[d],  e))
            end
        end

        R1 = P == Pr ? R - 1 : R + 1
        if 0 <= R1 <= D
            b = boundary(Val(P), Val(R), mfd)
            for f in funs
                bf = b * f
                bf::Fun{D,P,R1,T,T}
            end
        end

        R1 = P == Pr ? R + 1 : R - 1
        if 0 <= R1 <= D
            d = deriv(Val(P), Val(R), mfd)
            for f in funs
                df = d * f
                df::Fun{D,P,R1,T,T}
            end
        end
    end
end
