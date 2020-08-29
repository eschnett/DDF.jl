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
    @assert 0 < R <= D
    return Op{D,Pr,R - 1,Pr,R}(manifold, manifold.boundaries[R].op)
end
function boundary(::Val{Dl}, ::Val{R}, manifold::Manifold{D}) where {R,D}
    @assert 0 <= R < D
    return Op{D,Dl,R + 1,Dl,R}(manifold, manifold.boundaries[R + 1].op')
end

boundary(f::Fun{D,P,R}) where {D,P,R} = boundary(Val(P), Val(R), f.manifold) * f

# Derivative (coboundary)

export deriv
function deriv(::Val{P}, ::Val{R}, manifold::Manifold{D}) where {P,R,D}
    P::PrimalDual
    s = P == Pr ? +1 : -1
    @assert 0 <= R <= D
    @assert 0 <= R + s <= D
    return adjoint(boundary(Val(P), Val(R + s), manifold))::Op{D,P,R + s,P,R}
end

deriv(f::Fun{D,P,R}) where {D,P,R} = deriv(Val(P), Val(R), f.manifold) * f

################################################################################

# Geometric operators

# Hodge dual (circumcentric (diagonal) hodge operator)

export hodge, ⋆
function Forms.hodge(::Val{Pr}, ::Val{R},
                     manifold::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 <= R <= D
    vol = manifold.volumes[R]
    dualvol = manifold.dualvolumes[R]
    if isempty(vol)
        return zero(Op{D,Dl,R,Pr,R,S}, manifold)
    end
    return Op{D,Dl,R,Pr,R,S}(manifold, Diagonal(dualvol ./ vol))
end
function Forms.hodge(::Val{Dl}, ::Val{R},
                     manifold::Manifold{D,C,S}) where {R,D,C,S}
    @assert 0 <= R <= D
    vol = manifold.volumes[R]
    dualvol = manifold.dualvolumes[R]
    if isempty(vol)
        return zero(Op{D,Pr,R,Dl,R,S}, manifold)
    end
    return Op{D,Pr,R,Dl,R,S}(manifold, Diagonal(vol ./ dualvol))
end

export invhodge
function Forms.invhodge(::Val{P}, ::Val{R},
                        manifold::Manifold{D,C,S}) where {P,R,D,C,S}
    R::Int
    @assert 0 <= R <= D
    op = hodge(Val(!P), Val(R), manifold)
    return op::Op{D,P,R,!P,R,S}
end

Forms.hodge(f::Fun{D,P,R}) where {D,P,R} = hodge(Val(P), Val(R), f.manifold) * f

# Derivatives

export coderiv
function coderiv(::Val{P}, ::Val{R},
                 manifold::Manifold{D,C,S}) where {P,R,D,C,S}
    P::PrimalDual
    s = P == Pr ? +1 : -1
    @assert 0 <= R <= D
    @assert 0 <= R - s <= D
    return (bitsign(R) *
            hodge(Val(!P), Val(R - s), manifold) *
            deriv(Val(!P), Val(R), manifold) *
            hodge(Val(P), Val(R), manifold))::Op{D,P,R - s,P,R,S}
end

coderiv(f::Fun{D,P,R}) where {D,P,R} = coderiv(Val(P), Val(R), f.manifold) * f

export laplace
function laplace(::Val{P}, ::Val{R},
                 manifold::Manifold{D,C,S}) where {P,R,D,C,S}
    P::PrimalDual
    s = P == Pr ? +1 : -1
    @assert 0 <= R <= D
    op = zero(Op{D,P,R,P,R,S}, manifold)
    if 0 <= R - s <= D
        op += deriv(Val(P), Val(R - s), manifold) *
              coderiv(Val(P), Val(R), manifold)
    end
    if 0 <= R + s <= D
        op += coderiv(Val(P), Val(R + s), manifold) *
              deriv(Val(P), Val(R), manifold)
    end
    return op::Op{D,P,R,P,R,S}
end

laplace(f::Fun{D,P,R}) where {D,P,R} = laplace(Val(P), Val(R), f.manifold) * f

end
