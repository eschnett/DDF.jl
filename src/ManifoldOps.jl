module ManifoldOps

using ..Manifolds
using ..Ops

################################################################################

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

end
