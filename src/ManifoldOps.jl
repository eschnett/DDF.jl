module ManifoldOps

using DifferentialForms: Forms, hodge, invhodge
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

# Derivative (coboundary)

export deriv
function deriv(::Val{Pr}, ::Val{R}, manifold::Manifold{D}) where {R,D}
    @assert 0 <= R < D
    return adjoint(boundary(Val(Pr), Val(R + 1), manifold))::Op{D,Pr,R + 1,Pr,R}
end
function deriv(::Val{Dl}, ::Val{R}, manifold::Manifold{D}) where {R,D}
    @assert 0 < R <= D
    return adjoint(boundary(Val(Dl), Val(R - 1), manifold))::Op{D,Dl,R - 1,Dl,R}
end

################################################################################

# Geometric operators

# Hodge dual

# Circumcentric (diagonal) hodge operator
export hodge
function Forms.hodge(::Val{Pr}, ::Val{R}, manifold::Manifold{D,S}) where {R,D,S}
    D::Int
    R::Int
    @assert 0 <= R <= D

    vol = manifold.volumes[R]
    dualvol = manifold.dualvolumes[R]
    if isempty(vol)
        return zero(Op{D,Dl,R,Pr,R,S}, manifold)
    end
    return Op{D,Dl,R,Pr,R,S}(manifold, Diagonal(dualvol ./ vol))
end

function Forms.hodge(::Val{Dl}, ::Val{R}, manifold::Manifold{D,S}) where {R,D,S}
    D::Int
    R::Int
    @assert 0 <= R <= D

    vol = manifold.volumes[R]
    dualvol = manifold.dualvolumes[R]
    if isempty(vol)
        return zero(Op{D,Pr,R,Dl,R,S}, manifold)
    end
    return Op{D,Pr,R,Dl,R,S}(manifold, Diagonal(vol ./ dualvol))
end

export invhodge
function Forms.invhodge(::Val{P}, ::Val{R}, manifold::Manifold) where {P,R}
    return hodge(Val(!P), Val(R), manifold)
end

Forms.hodge(f::Fun{D,P,R}) where {D,P,R} = hodge(Val(P), Val(R), f.manifold) * f

# # Derivatives
# 
# export coderiv
# function coderiv(::Val{Pr}, ::Val{R}, geom::Geometry{D,T}) where {R,D,T}
#     D::Int
#     T::Type
#     @assert 0 < R <= D
#     op = hodge(Val(Dl), Val(R - 1), geom) *
#          deriv(Val(Dl), Val(R), geom.topo) *
#          hodge(Val(Pr), Val(R), geom)
#     return op::Op{D,Pr,R - 1,Pr,R,T}
# end
# 
# export laplace
# function laplace(::Val{Pr}, ::Val{R}, geom::Geometry{D,T}) where {R,D,T}
#     D::Int
#     T::Type
#     @assert 0 <= R <= D
#     op = zero(Op{D,Pr,R,Pr,R,T}, geom.topo)
#     if R > 0
#         op += deriv(Val(Pr), Val(R - 1), geom.topo) *
#               coderiv(Val(Pr), Val(R), geom)
#     end
#     if R < D
#         op += coderiv(Val(Pr), Val(R + 1), geom) *
#               deriv(Val(Pr), Val(R), geom.topo)
#     end
#     return op::Op{D,Pr,R,Pr,R,T}
# end

end
