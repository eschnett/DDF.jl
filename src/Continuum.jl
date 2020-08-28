module Continuum

using Bernstein
using ComputedFieldTypes
using DifferentialForms
using NearestNeighbors
using SimplexQuad
using StaticArrays

using ..Funs
using ..Manifolds
using ..SparseOps

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
    lookup = mfd.lookup[(D, 0)]
    for j in sparse_column_rows(lookup, i)
        sj = sparse_column_rows(mfd.simplices[D], j)
        @assert length(sj) == N
        sj = SVector{D + 1,Int}(sj[n] for n in 1:N)
        # Coordinates of simplex vertices
        # This is only correct for D == 0, P == Pr
        @assert D == 0 && P == Pr
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

export project
"""
Project a function onto a manifold
"""
function project(::Type{<:Fun{D,P,R,S,T}}, f,
                 mfd::Manifold{D,S}) where {D,P,R,S,T}
    @assert P == Pr             # TODO
    @assert R == 0              # TODO

    if D == 0
        values = T[f(SVector{D,S}())[]]
        return Fun{D,P,R,S,T}(mfd, values)
    end

    order = 4                   # Choice

    N = D + 1

    f(zero(SVector{D,S}))::Form{D,R,T}

    values = Array{T}(undef, nsimplices(mfd, R))
    # Loop over all R-simplices
    for i in 1:nsimplices(mfd, R)
        value = zero(T)
        # Loop over the support of the R-simplex's basis functions,
        # i.e. all neighbouring top simplices
        for j in sparse_column_rows(mfd.lookup[(D, R)], i)
            sj = sparse_column_rows(mfd.simplices[D], j)
            @assert length(sj) == N
            sj = SVector{N,Int}(sj[n] for n in 1:N)
            # Coordinates of simplex vertices
            xs = SVector{N,SVector{D,S}}(SVector{D,S}(@view mfd.coords[k, :])
                                         for k in sj)
            xs::SVector{N,SVector{D,S}}
            setup = cartesian2barycentric_setup(xs)

            # Find integration method
            XS = S[xs[n][d] for n in 1:N, d in 1:D]
            XS::Array{S,2}
            X, W = simplexquad(order, [xs[n][d] for n in 1:N, d in 1:D])

            # `findfirst` here works only for D == 0
            @assert D == 0
            n = findfirst(==(i), sj)
            @assert n !== nothing
            # `[]` here works only for D == 0
            @assert D == 0
            kernel(x::SVector{D,T}) = basis_x(setup, n, x) * f(x)[]

            bf = integrate_x(kernel, Val(D), X, W)::T
            value += bf
        end
        values[i] = value
    end
    @error ""
    return Fun{D,P,R,S,T}(mfd, values)
end

################################################################################

# TODO: Remove this, use Bernstein polynomials instead
@fastmath function basis_λ(n::Int, λ::SVector{N,S}) where {N,S}
    N::Int
    @assert 1 <= n <= N
    return λ[n]
end

@fastmath function integrate_x(::Val{D}, f, X, W) where {D}
    D::Int
    @assert D > 0
    @assert size(X, 2) == D
    @assert size(X, 1) == size(W, 1)

    @inbounds begin
        s = zero(W[1]) * f(SVector{D}(@view X[1, :]))
        for n in 1:length(W)
            s += W[n] * f(SVector{D}(@view X[n, :]))
        end
        return s
    end
end

end
