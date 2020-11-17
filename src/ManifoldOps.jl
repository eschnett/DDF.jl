module ManifoldOps

using DifferentialForms: Forms, bitsign, hodge, invhodge, ⋆
using LinearAlgebra

using ..Funs
using ..Manifolds
using ..Ops

################################################################################

# Topological operators

# Boundary

export boundary
function boundary(::Val{Pr}, ::Val{R}, manifold::Manifold{D}) where {R,D}
    @assert 0 < R ≤ D
    # println("[boundary(D=$D, P=Pr, R=$R)]")
    return Op{D,Pr,R - 1,Pr,R}(manifold, get_boundaries(manifold, R).op)
end
function boundary(::Val{Dl}, ::Val{R}, manifold::Manifold{D}) where {R,D}
    @assert 0 ≤ R < D
    # println("[boundary(D=$D, P=Dl, R=$R)]")
    return Op{D,Dl,R + 1,Dl,R}(manifold, get_boundaries(manifold, R + 1).op')
end

boundary(f::Fun{D,P,R}) where {D,P,R} = boundary(Val(P), Val(R), f.manifold) * f

export isboundary
function isboundary(::Val{Pr}, ::Val{R}, manifold::Manifold{D}) where {R,D}
    @assert 0 ≤ R < D
    # println("[isboundary(D=$D, P=Pr, R=$R)]")
    return Op{D,Pr,R,Pr,R}(manifold, map(Bool, get_isboundary(manifold, R)).op)
end

# Derivative (coboundary)

export deriv
function deriv(::Val{P}, ::Val{R}, manifold::Manifold{D}) where {P,R,D}
    P::PrimalDual
    s = P == Pr ? +1 : -1
    @assert 0 ≤ R ≤ D
    @assert 0 ≤ R + s ≤ D
    # println("[deriv(D=$D, P=$P, R=$R)]")
    return adjoint(boundary(Val(P), Val(R + s), manifold))::Op{D,P,R + s,P,R}
end

deriv(f::Fun{D,P,R}) where {D,P,R} = deriv(Val(P), Val(R), f.manifold) * f

################################################################################

# Geometric operators

# Hodge dual (circumcentric (diagonal) hodge operator)

export hodge, ⋆
function Forms.hodge(::Val{Pr}, ::Val{R},
                     manifold::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R ≤ D
    # println("[hodge(D=$D, P=Pr, R=$R)]")
    vol = get_volumes(manifold, R)
    dualvol = get_dualvolumes(manifold, R)
    if isempty(vol)
        return zero(Op{D,Dl,R,Pr,R,S}, manifold)
    end
    return Op{D,Dl,R,Pr,R,S}(manifold, Diagonal(dualvol ./ vol))
end

export invhodge
function Forms.invhodge(::Val{Dl}, ::Val{R},
                        manifold::Manifold{D,C,S}) where {P,R,D,C,S}
    R::Int
    @assert 0 ≤ R ≤ D
    # println("[invhodge(D=$D, P=Dl, R=$R)]")
    vol = get_volumes(manifold, R)
    dualvol = get_dualvolumes(manifold, R)
    if isempty(vol)
        return zero(Op{D,Pr,R,Dl,R,S}, manifold)
    end
    return Op{D,Pr,R,Dl,R,S}(manifold, Diagonal(vol ./ dualvol))
end

function Forms.hodge(::Val{Dl}, ::Val{R},
                     manifold::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R ≤ D
    s = bitsign(R * (D - R))
    return s * invhodge(Val(Dl), Val(R), manifold)
end

function Forms.invhodge(::Val{Pr}, ::Val{R},
                        manifold::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 ≤ R ≤ D
    s = bitsign(R * (D - R))
    return s * hodge(Val(Pr), Val(R), manifold)
end

Forms.hodge(f::Fun{D,P,R}) where {D,P,R} = hodge(Val(P), Val(R), f.manifold) * f
function Forms.invhodge(f::Fun{D,P,R}) where {D,P,R}
    return invhodge(Val(P), Val(R), f.manifold) * f
end

# Derivatives

export coderiv
function coderiv(::Val{P}, ::Val{R},
                 manifold::Manifold{D,C,S}) where {P,R,D,C,S}
    P::PrimalDual
    dR = P == Pr ? +1 : -1
    @assert 0 ≤ R ≤ D
    @assert 0 ≤ R - dR ≤ D
    # println("[coderiv(D=$D, P=$P, R=$R)]")
    return (bitsign(R) *
            hodge(Val(!P), Val(R - dR), manifold) *
            deriv(Val(!P), Val(R), manifold) *
            hodge(Val(P), Val(R), manifold))::Op{D,P,R - dR,P,R,S}
end

coderiv(f::Fun{D,P,R}) where {D,P,R} = coderiv(Val(P), Val(R), f.manifold) * f

# Δ
export laplace
# TODO: memoise laplace operator (and hodge, coderiv, etc.)
function laplace(::Val{P}, ::Val{R},
                 manifold::Manifold{D,C,S}) where {P,R,D,C,S}
    P::PrimalDual
    @assert 0 ≤ R ≤ D
    dR = P == Pr ? +1 : -1
    # println("[laplace(D=$D, P=$P, R=$R)]")
    op = zero(Op{D,P,R,P,R,S}, manifold)
    if 0 ≤ R - dR ≤ D
        op += deriv(Val(P), Val(R - dR), manifold) *
              coderiv(Val(P), Val(R), manifold)
    end
    if 0 ≤ R + dR ≤ D
        op += coderiv(Val(P), Val(R + dR), manifold) *
              deriv(Val(P), Val(R), manifold)
    end
    return op::Op{D,P,R,P,R,S}
end

laplace(f::Fun{D,P,R}) where {D,P,R} = laplace(Val(P), Val(R), f.manifold) * f

# ∫
export integral
integral(f::Fun{D,Pr,D}) where {D} = sum(f.values)
integral(f::Fun{D,Dl,0}) where {D} = sum(f.values)

# ⋅
export dot
function LinearAlgebra.dot(f::Fun{D,Pr,D}, g::Fun{D,Pr,D}) where {D}
    @assert f.manifold ≡ g.manifold
    vol = get_volumes(f.manifold, D)
    return dot(f.values, Diagonal(1 ./ vol), g.values)
end
function LinearAlgebra.dot(f::Fun{D,Dl,0}, g::Fun{D,Dl,0}) where {D}
    @assert f.manifold ≡ g.manifold
    dualvol = get_dualvolumes(f.manifold, 0)
    return dot(f.values, Diagonal(1 ./ dualvol), g.values)
end

# LinearAlgebra.norm(f::Fun) = norm(f.values)
# LinearAlgebra.norm(f::Fun, p::Real) = norm(f.values, p)

end
