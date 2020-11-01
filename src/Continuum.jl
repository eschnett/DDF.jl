module Continuum

using Bernstein
using Combinatorics
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
    mfd = f.manifold

    # Find nearest vertex
    if D == 0
        y = f.values[1]
        return Form{D,R,T}((y,))
    end

    i, dist = nn(simplex_tree(mfd), x)
    # Search all neighbouring simplices to find containing simplex
    lookup_D = lookup(mfd, D, 0)::SparseOp{D,0,One}
    lookup_R = lookup(mfd, R, D)::SparseOp{R,D,One}
    for j in sparse_column_rows(lookup_D, i)
        sj = sparse_column_rows(mfd.simplices[D], j)
        sj = SVector{D + 1,Int}(sj[n] for n in 1:(D + 1))
        # Coordinates of simplex vertices
        xs = SVector{D + 1,SVector{C,S}}(coords(mfd)[k] for k in sj)
        # Calculate barycentric coordinates
        x2λ = cartesian2barycentric_setup(xs)
        λ = cartesian2barycentric(x2λ, x)
        # delta = S(0)
        delta = 10 * eps(S)
        if all(λi -> -delta ≤ λi ≤ 1 + delta, λ)
            # Found simplex

            @assert P == Pr
            sk = sparse_column_rows(lookup_R, j)
            N = DifferentialForms.Forms.binomial(Val(D + 1), Val(R + 1))
            sk = SVector{N,Int}(sk[n] for n in 1:N)
            # Values at `R`-simplices
            ys = f.values[sk]

            # Evaluate basis functions
            dλ2dx = dbarycentric2dcartesian_setup(Form{D,R}, xs)
            y = zero(Form{D,R,T})
            for n in 1:N
                y += map(b -> ys[n] * b, basis_x(Form{D,R}, x2λ, dλ2dx, n, x))
            end
            return y
        end
    end
    return error("Coordinate $x not found ∈ manifold $(mfd.name)")
end

################################################################################

export sample
"""
Sample a function on a manifold
"""
function sample(::Type{<:Fun{D,P,R,C,S,T}}, f,
                mfd::Manifold{D,C,S}) where {D,P,R,C,S,T}
    @assert P == Pr             # TODO
    lookup_D = lookup(mfd, D, R)::SparseOp{D,R,One}
    simplices_D = mfd.simplices[D]::SparseOp{0,D,One}
    simplices_R = mfd.simplices[R]::SparseOp{0,R,One}
    values = Array{T}(undef, nsimplices(mfd, R))
    for i in 1:nsimplices(mfd, R)
        @assert P == Pr

        sj = sparse_column_rows(lookup_D, i)
        j = sj[1]               # pick a simplex at random
        # Coordinates of simplex vertices
        sk = sparse_column_rows(simplices_D, j)
        xs = SVector{D + 1,SVector{C,S}}(coords(mfd)[k] for k in sk)
        # Evaluate basis function
        x2λ = cartesian2barycentric_setup(xs)
        dλ2dx = dbarycentric2dcartesian_setup(Form{D,R}, xs)

        # sj = sparse_column_rows(simplices_R, i)
        # # Coordinates of simplex vertices
        # xs = SVector{R + 1,SVector{C,S}}(coords(mfd)[k] for k in sj)
        # # Evaluate basis function
        # x2λ = cartesian2barycentric_setup(xs)
        # dλ2dx = dbarycentric2dcartesian_setup(Form{D,R}, xs)

        si = sparse_column_rows(simplices_R, i)
        bits = zero(SVector{D + 1,Bool})
        @assert length(sk) == D + 1
        for i′ in si
            # `findfirst` returns the key, i.e. the row number
            # n = findfirst(==(i′), sk)
            n = nothing
            for (n′, k′) in enumerate(sk)
                if k′ == i′
                    n = n′
                    break
                end
            end
            @assert n ≢ nothing
            bits = Base.setindex(bits, true, n)
        end
        @assert count(bits) == R + 1
        n = DifferentialForms.Forms.bit2lin(Val(D + 1), Val(R + 1), bits)

        x = coords(mfd, R)[i]
        b = basis_x(Form{D,R}, x2λ, dλ2dx, n, x)::Form{D,R,S}
        y = f(x)::Form{D,R,T}

        values[i] = (b ⋅ y)[]::T / norm2(b)
        @assert isfinite(values[i])
    end
    return Fun{D,P,R,C,S,T}(mfd, values)
end

################################################################################

export project
"""
Project a function onto a manifold
"""
function project(::Type{<:Fun{D,P,R,C,S,T}}, f,
                 mfd::Manifold{D,C,S}) where {D,P,R,C,S,T}
    D::Int
    R::Int
    C::Int
    @assert 0 ≤ R ≤ D ≤ C
    @assert C == D
    N = D + 1
    @assert P == Pr             # TODO

    if D == 0
        value = f(SVector{C,S}())[]
        return Fun{D,P,R,C,S,T}(mfd, T[value])
    end

    order = 4                   # Choice
    scheme = integration_scheme(S, Val(D), Val(order))

    B = basis_products(Val(P), Val(R), mfd)

    f(zero(SVector{C,S}))::Form{D,R,T}

    values = zeros(T, nsimplices(mfd, R))
    # Loop over all D-simplices
    for i in 1:nsimplices(mfd, D)
        si = sparse_column_rows(mfd.simplices[D]::SparseOp{0,D,One}, i)
        @assert length(si) == N
        si = SVector{N,Int}(si[n] for n in 1:N)

        # Coordinates of simplex vertices
        xs = SVector{N,SVector{C,S}}(coords(mfd)[n] for n in si)
        x2λ = cartesian2barycentric_setup(xs)
        dλ2dx = dbarycentric2dcartesian_setup(Form{D,R}, xs)

        # Loop over all contained R-simplices
        lookup_RD = lookup(mfd, R, D)::SparseOp{R,D,One}
        iter = enumerate(sparse_column_rows(lookup_RD, i))
        for (nj, j) in iter
            function kernel(x::SVector{C,S})
                bj = basis_x(Form{D,R}, x2λ, dλ2dx, nj, x)
                return (bj ⋅ f(x))[]
            end
            bf = integrate(kernel, scheme, xs)
            values[j] += bf
        end
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
    @assert 0 ≤ R ≤ D ≤ C
    @assert C == D
    N = D + 1

    # `order=2` suffices because we only have linear basis functions
    order = 2
    scheme = integration_scheme(S, Val(D), Val(order))

    # Result: sparse matrix
    I = Int[]
    J = Int[]
    V = S[]

    # Loop over all D-simplices
    for i in 1:nsimplices(mfd, D)
        si = sparse_column_rows(mfd.simplices[D]::SparseOp{0,D,One}, i)
        @assert length(si) == N
        si = SVector{N,Int}(si[n] for n in 1:N)

        # Coordinates of simplex vertices
        xs = SVector{N,SVector{C,S}}(coords(mfd)[n] for n in si)
        x2λ = cartesian2barycentric_setup(xs)
        dλ2dx = dbarycentric2dcartesian_setup(Form{D,R}, xs)

        # Double loop over all contained R-simplices
        lookup_RD = lookup(mfd, R, D)::SparseOp{R,D,One}
        iter = enumerate(sparse_column_rows(lookup_RD, i))
        for (nj, j) in iter
            for (nk, k) in iter
                function kernel(x::SVector{C,S})
                    bj = basis_x(Form{D,R}, x2λ, dλ2dx, nj, x)
                    bk = basis_x(Form{D,R}, x2λ, dλ2dx, nk, x)
                    return (bj ⋅ bk)[]
                end
                b = integrate(kernel, scheme, xs)

                push!(I, j)
                push!(J, k)
                push!(V, b)
            end
        end
    end

    n = nsimplices(mfd, R)
    return sparse(I, J, V, n, n)
end

################################################################################

# @fastmath function basis_x(setup, n::Int, x::SVector{C,T}) where {C,T}
#     λ = cartesian2barycentric(setup, x)
#     return basis_λ(n, λ)
# end
# 
# # TODO: Remove this, use Bernstein polynomials instead
# @fastmath function basis_λ(n::Int, λ::SVector{N,S}) where {N,S}
#     @assert 1 ≤ n ≤ N
#     return λ[n]
# end

# See: Douglas Arnold, Richard Falk, Ragnar Winther, "Finite element
# exterior calculus, homological techniques, and applications", Acta
# Numerica 15, 1-155 (2006), DOI:10.1017/S0962492906210018, section
# 3.3
#
# P-(r) Λ(k): reduced space of polynomial `k`-forms with order up to `r`
# P-(r) Λ(k) = P(r-1) Λ(k) ⊕ κ H(r-1) Λ(k+1)

# D     R     [n]
# 2     0     1   x   y
# 2     1     dx   dy   x dy - y dx
# 2     2     dx∧dy

# D     R     [n]
# 3     0     1   x   y   z
# 3     1     dx   dy   dz   x dy - y dx   x dz - z dx   y dz - z dy
# 3     2     dx∧dy   dx∧dz   dy∧dz   x dy dz - y dx dz + z dx dy
# 3     3     dx∧dy∧dz

# Expressed ∈ terms of barycentric coordinates:

# x = xⁱ λᵢ = x¹ λ₁ + x² λ₂ + x³ λ₃
# y = yⁱ λᵢ = y¹ λ₁ + y² λ₂ + y³ λ₃
# 1 = Σᵢ λᵢ = λ₁ + λ₂ + λ₃
# dx = xⁱ dλᵢ = x¹ dλ₁ + x² dλ₂ + x³ dλ₃
# dy = yⁱ dλᵢ = y¹ dλ₁ + y² dλ₂ + y³ dλ₃
# 0  = Σᵢ dλᵢ = dλ₁ + dλ₂ + dλ₃

# D     R     [n]
#
# 1     0     λ₁   λ₂
# 1     1     (λ₁ + λ₂) (- dλ₁ + dλ₂)
#
# 2     0     λ₁   λ₂   λ₃
# 2     1     (λ₁ + λ₂) (- dλ₁ + dλ₂)   (λ₁ + λ₃) (- dλ₁ + dλ₃)
#             (λ₂ + λ₃) (- dλ₂ + dλ₃)
# 2     2     (λ₁ + λ₂ + λ₃) (dλ₁₂ - dλ₁₃ + dλ₂₃)

# To determine signs:
# 1. look at vertices that make up the face
# 2. commute the derivative index to the left; that determines the parity

#
# 3     0     λ₁   λ₂   λ₃   λ₄
# 3     1     (λ₁ + λ₂) (- dλ₁ + dλ₂)   (λ₁ + λ₃) (- dλ₁ + dλ₃)
#             (λ₁ + λ₄) (- dλ₁ + dλ₄)   (λ₂ + λ₃) (- dλ₂ + dλ₃)
#             (λ₂ + λ₄) (- dλ₂ + dλ₄)   (λ₃ + λ₄) (- dλ₃ + dλ₄)
# 3     2     (λ₁ + λ₂ + λ₃) (dλ₁₂ - dλ₁₃ + dλ₂₃)
#             (λ₁ + λ₂ + λ₄) (dλ₁₂ - dλ₁₄ + dλ₂₄)
#             (λ₁ + λ₃ + λ₄) (dλ₁₃ - dλ₁₄ + dλ₃₄)
#             (λ₂ + λ₃ + λ₄) (dλ₂₃ - dλ₂₄ + dλ₃₄)
# 3     3     (λ₁ + λ₂ + λ₃ + λ₄) (- dλ₁₂₃ + dλ₁₂₄ - dλ₁₃₄ + dλ₂₃₄)

# xᵢ = xⁿᵢ λₙ
# dxᵢ = xⁿᵢ dλₙ

# λₙ = Xⁱₙ xᵢ + bₙ
# dλₙ = Xⁱₙ dxᵢ

# dλ₁ + dλ₂                     = dλ₁ - dλ₁                     = 0
# dλ₁₂ + dλ₁₃ + dλ₂₃            = dλ₁₂ - dλ₁₂ + dλ₁₂            = dλ₁₂
# dλ₁₂₃ + dλ₁₂₄ + dλ₁₃₄ + dλ₂₃₄ = dλ₁₂₃ - dλ₁₂₃ + dλ₁₂₃ + dλ₁₂₃ = 2 dλ₁₂₃

# dx1 = x1i dλi
# dx1 ∧ dx2 = x1i dλi ∧ x2j dλj = x1i x2j dλi ∧ dλj
# dx1 ∧ dx2 ∧ dx3 = x1i x2j x3k dλi ∧ dλj ∧ dλk

# x[i] = xs[n][i] λ[n]
# dx[i] = xs[n][i] dλ[n]
#
# dx[i] ∧ dx[j] = xs[n][i] dλ[n] ∧ xs[m][j] dλ[m]
#               = xs[n][i] xs[m][j] dλ[n] ∧ dλ[m]

################################################################################

function basis_x(::Type{<:Form{D,R}}, x2λ, dλ2dx, n::Int,
                 x::SVector{C,T}) where {D,R,C,T}
    λ = cartesian2barycentric(x2λ, x)
    bλ = basis_λ(Form{D,R}, n, λ)::Form{D + 1,R,T}
    bx = dbarycentric2dcartesian(dλ2dx, bλ)::Form{D,R,T}
    return bx
end

# @fastmath
function dbarycentric2dcartesian(dλ2dx::SMatrix{Nx,Nλ},
                                 dλ::Form{N,R}) where {Nx,Nλ,N,R}
    N::Int
    @assert N ≥ 1
    D = N - 1
    R::Int
    @assert 0 ≤ R ≤ D
    @assert Nx == length(Form{D,R})
    @assert Nλ == length(Form{D + 1,R})
    return Form{D,R}(dλ2dx * dλ.elts)
end

#TODO @fastmath
function dbarycentric2dcartesian_setup(::Type{<:Form{D,R}},
                                       xs::SVector{N,SVector{C,T}}) where {D,R,
                                                                           N,C,
                                                                           T}
    # Form dimension and rank
    D::Int
    R::Int
    @assert 0 ≤ R ≤ D
    N::Int
    @assert N == D + 1
    C::Int
    @assert C == D
    # Number of barycentric form elements
    Nλ = length(Form{D + 1,R})
    # Number of physical form elements
    Nx = length(Form{D,R})

    r = zero(SMatrix{Nx,Nλ,T})
    for nx in 1:Nx, nλ in 1:Nλ
        lstx = DifferentialForms.Forms.lin2lst(Val(D), Val(R), nx)
        lstλ = DifferentialForms.Forms.lin2lst(Val(D + 1), Val(R), nλ)
        relt = zero(T)
        lstλs = permutations(lstλ)
        for lstλ′ in lstλs
            rterm = one(T)
            _, p = sort_perm(lstλ′)
            rterm *= bitsign(p)
            for (lx, lλ) in zip(lstx, lstλ′)
                rterm *= xs[lλ][lx]
            end
            relt += rterm
        end
        r = setindex(r, relt, nx, nλ)
    end
    return r
end

#TODO @fastmath
function basis_λ(::Type{<:Form{D,R}}, i::Int, λ::SVector{N,T}) where {D,R,C,N,T}
    D::Int
    R::Int
    @assert 0 ≤ R ≤ D
    # Number of barycentric coordinates
    @assert N == D + 1
    # Number of barycentric form elements
    Nλ = length(Form{D + 1,R})
    # Number of basis functions
    I = DifferentialForms.binomial(Val(D + 1), Val(R + 1))
    @assert 1 ≤ i ≤ I

    # Find functional dependency on λᵢ
    inds = DifferentialForms.Forms.lin2lst(Val(D + 1), Val(R + 1), i)
    rval = sum(λ[inds])

    # Find contributing components and their parity
    bits = DifferentialForms.Forms.lin2bit(Val(D + 1), Val(R + 1), i)
    relts = zero(SVector{Nλ,T})
    # TODO: Can this be implemented without this loop?
    for ind in inds
        @assert bits[ind]
        bits2 = Base.setindex(bits, false, ind)
        p = count(bits[1:(ind - 1)])
        s = bitsign(p)
        j = DifferentialForms.Forms.bit2lin(Val(D + 1), Val(R), bits2)
        relts = setindex(relts, relts[j] + s * rval, j)
    end

    return Form{D + 1,R}(relts) / factorial(R)
end

@generated function integration_scheme(::Type{T}, ::Val{D},
                                       ::Val{order}) where {T,D,order}
    return grundmann_moeller(T, Val(D), order + iseven(order))
end

end
