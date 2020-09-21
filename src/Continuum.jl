module Continuum

using Bernstein
using ComputedFieldTypes
using DifferentialForms
using GrundmannMoeller
using NearestNeighbors
using SparseArrays
using StaticArrays

using ..Funs
using ..Manifolds
using ..SparseOps
using ..ZeroOrOne

################################################################################

export evaluate
"""
Evaluate a function at a point
"""
function evaluate(f::Fun{D,P,R,C,S,T}, x::SVector{C,S}) where {D,P,R,C,S,T}
    @assert C == D
    N = D + 1
    mfd = f.manifold

    # Find nearest vertex
    if D == 0
        y = f.values[1]
        return Form{D,R,T}((y,))
    end

    i, dist = nn(mfd.simplex_tree, x)
    # Search all neighbouring simplices to find containing simplex
    lookup = mfd.lookup[(D, 0)]
    for j in sparse_column_rows(lookup, i)
        sj = sparse_column_rows(mfd.simplices[D], j)
        @assert length(sj) == N
        sj = SVector{D + 1,Int}(sj[n] for n in 1:N)
        # Coordinates of simplex vertices
        # This is only correct for P == Pr, R == 0
        @assert P == Pr && R == 0
        xs = SVector{N,SVector{C,S}}(mfd.coords[0][k] for k in sj)
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
    error("Coordinate $x not found in manifold $(mfd.name)")
end

################################################################################

export sample
"""
Sample a function on a manifold
"""
function sample(::Type{<:Fun{D,P,R,C,S,T}}, f,
                mfd::Manifold{D,C,S}) where {D,P,R,C,S,T}
    @assert P == Pr             # TODO
    @assert R == 0              # TODO
    values = Array{T}(undef, nsimplices(mfd, R))
    for i in 1:nsimplices(mfd, R)
        x = mfd.coords[0][i]
        y = f(x)
        y::Form{D,R,T}
        values[i] = y[]::T
    end
    return Fun{D,P,R,C,S,T}(mfd, values)
end

################################################################################

@generated function integration_scheme(::Type{T}, ::Val{D},
                                       ::Val{order}) where {T,D,order}
    grundmann_moeller(T, Val(D), order + iseven(order))
end

export project
"""
Project a function onto a manifold
"""
function project(::Type{<:Fun{D,P,R,C,S,T}}, f,
                 mfd::Manifold{D,C,S}) where {D,P,R,C,S,T}
    @assert P == Pr             # TODO
    @assert R == 0              # TODO
    N = D + 1
    @assert C == D

    if D == 0
        values = T[f(SVector{C,S}())[]]
        return Fun{D,P,R,C,S,T}(mfd, values)
    end

    order = 4                   # Choice
    scheme = integration_scheme(S, Val(D), Val(order))

    B = basis_products(Val(Pr), Val(R), mfd)

    f(zero(SVector{C,S}))::Form{D,R,T}

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
            xs = SVector{N,SVector{C,S}}(mfd.coords[0][k] for k in sj)
            setup = cartesian2barycentric_setup(xs)

            # `findfirst` here works only for R == 0
            @assert R == 0
            n = findfirst(==(i), sj)
            @assert n !== nothing
            # `[]` here works only for R == 0
            @assert R == 0
            kernel(x::SVector{C,T}) = basis_x(setup, n, x) * f(x)[]

            bf = integrate(kernel, scheme, xs)

            value += bf
        end
        values[i] = value
    end

    values′ = B \ values

    return Fun{D,P,R,C,S,T}(mfd, values′)
end

################################################################################

function basis_products(::Val{Pr}, ::Val{R},
                        mfd::Manifold{D,C,S}) where {D,R,C,S}
    # Check arguments
    D::Int
    R::Int
    C::Int
    @assert 0 <= R <= D <= C
    @assert C == D
    N = D + 1

    # `order=2` suffices because we only have linear basis functions
    order = 2
    scheme = integration_scheme(S, Val(D), Val(order))

    # Result: sparse matrix
    I = Int[]
    J = Int[]
    V = S[]

    # Loop over all R-simplices
    for i in 1:nsimplices(mfd, R)
        # si = sparse_column_rows(mfd.simplices[R], i)
        # Loop over the support of this R-simplex's basis functions,
        # i.e. all neighbouring top simplices
        for k in sparse_column_rows(mfd.lookup[(D, R)]::SparseOp{D,R,One}, i)
            sk = sparse_column_rows(mfd.simplices[D], k)
            @assert length(sk) == N
            # Loop over all contributing other basis functions, i.e.
            # all neighbouring R-simplices
            for j in
                sparse_column_rows(mfd.lookup[(R, D)]::SparseOp{R,D,One}, k)
                # sj = sparse_column_rows(mfd.simplices[R], j)
                # @assert k ∈ sj
                # Coordinates of simplex vertices
                xs = SVector{N,SVector{C,S}}(mfd.coords[0][l] for l in sk)
                setup = cartesian2barycentric_setup(xs)

                # Find basis functions for simplices i and j
                @assert R == 0
                ni = findfirst(==(i), sk)
                nj = findfirst(==(j), sk)
                function kernel(x::SVector{C,S})
                    basis_x(setup, ni, x) * basis_x(setup, nj, x)
                end
                b = integrate(kernel, scheme, xs)

                push!(I, i)
                push!(J, j)
                push!(V, b)
            end
        end
    end

    n = nsimplices(mfd, R)
    return sparse(I, J, V, n, n)
end

################################################################################

@fastmath function basis_x(setup, n::Int, x::SVector{C,T}) where {C,T}
    C::Int
    λ = cartesian2barycentric(setup, x)
    return basis_λ(n, λ)
end

# TODO: Remove this, use Bernstein polynomials instead
@fastmath function basis_λ(n::Int, λ::SVector{N,S}) where {N,S}
    N::Int
    @assert 1 <= n <= N
    return λ[n]
end

end
