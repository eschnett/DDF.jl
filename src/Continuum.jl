module Continuum

using Bernstein
using ComputedFieldTypes
using DifferentialForms
using NearestNeighbors
using StaticArrays

using ..Funs
using ..Manifolds
using ..SparseOps

################################################################################

export sample
"""
Sample a function on a manifold
"""
function sample(::Type{<:Fun{D,P,R,S,T}}, f,
                mfd::Manifold{D,S}) where {D,P,R,S,T}
    @assert P == Pr             # TODO
    @assert R == 0              # TODO
    values = Array{T}(undef, nsimplices(mfd, R))
    for i in 1:nsimplices(mfd, R)
        x = SVector{D,S}(@view mfd.coords[i, :])
        y = f(x)
        y::Form{D,R,T}
        values[i] = y[]::T
    end
    return Fun{D,P,R,S,T}(mfd, values)
end

################################################################################

export evaluate
"""
Evaluate a function at a point
"""
function evaluate(f::Fun{D,P,R,S,T}, x::SVector{D,S}) where {D,P,R,S,T}
    N = D + 1
    mfd = f.manifold

    # Find nearest vertex
    if D == 0
        y = f.values[1]
        return Form{D,R,T}((y,))
    end

    mfd.simplex_tree::KDTree
    i, dist = nn(mfd.simplex_tree, x)
    # Search all neighbouring simplices to find containing simplex
    # lookup = mfd.lookup[(D,0)]
    lookup = mfd.simplices[D]'
    for j in sparse_column_rows(lookup, i)
        sj = sparse_column_rows(mfd.simplices[D], j)
        @assert length(sj) == N
        sj = SVector{D + 1,Int}(sj[n] for n in 1:N)

        # Coordinates of sjmplex vertices
        # xs = map(k -> SVector{D,S}(@view coords[k, :]) for k in sj)
        # xs::SVector{N,SVector{D,S}}
        xs = SVector{N,SVector{D,S}}(SVector{D,S}(@view mfd.coords[k, :])
                                     for k in sj)
        # setup = cartesian2barycentric_setup(xs)

        # Calculate barycentric coordinates
        λ = cartesian2barycentric(xs, x)
        # delta = S(0)
        delta = 10 * eps(S)
        if all(λi -> -delta <= λi <= 1 + delta, λ)
            ys = f.values[sj]
            # Linear interpolation
            # TODO: Use Bernstein polynomials instead
            y = sum(ys[n] * basis_λ(n, λ) for n in 1:N)
            return Form{D,R,T}((y,))
        end
    end
    @error "Coordinate $x not found in manifold $(mfd.name)"
end

# TODO: Remove this, use Bernstein polynomials instead
@fastmath function basis_λ(n::Int, λ::SVector{N,S}) where {N,S}
    N::Int
    @assert 1 <= n <= N
    return λ[n]
end

end
