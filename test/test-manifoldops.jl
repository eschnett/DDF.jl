using DDF

using StaticArrays
using Test

@testset "Manifold ops D=$D P=$P R=$R" for D in 0:Dmax, P in (Pr, Dl), R in 0:D
    S = Rational{Int128}

    mfds = [empty_manifold(Val(D), S), simplex_manifold(Val(D), S),
            hypercube_manifold(Val(D), S)]

    for mfd in mfds
        funs = Fun{D,P,R,S}[]
        push!(funs, zero(Fun{D,P,R,S,S}, mfd))
        e = id(Fun{D,P,0,S,SVector{D,S}}, mfd)
        if R == 0
            push!(funs, e)
        end

        R1 = P == Pr ? R - 1 : R + 1
        if 0 <= R1 <= D
            b = boundary(Val(P), Val(R), mfd)
            for f in funs
                bf = b * f
                T = eltype(f)
                bf::Fun{D,P,R1,S,T}
            end
        end

        R1 = P == Pr ? R + 1 : R - 1
        if 0 <= R1 <= D
            d = deriv(Val(P), Val(R), mfd)
            for f in funs
                df = d * f
                T = eltype(f)
                df::Fun{D,P,R1,S,T}
            end
        end
    end
end
