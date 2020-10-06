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
    mfd = f.manifold

    # Find nearest vertex
    if D == 0
        y = f.values[1]
        return Form{D,R,T}((y,))
    end

    i, dist = nn(mfd.simplex_tree, x)
    # Search all neighbouring simplices to find containing simplex
    lookup_D = mfd.lookup[(D, 0)]::SparseOp{D,0,One}
    lookup_R = mfd.lookup[(R, D)]::SparseOp{R,D,One}
    for j in sparse_column_rows(lookup_D, i)
        sj = sparse_column_rows(mfd.simplices[D], j)
        sj = SVector{D + 1,Int}(sj[n] for n in 1:(D + 1))
        # Coordinates of simplex vertices
        xs = SVector{D + 1,SVector{C,S}}(mfd.coords[0][k] for k in sj)
        # Calculate barycentric coordinates
        x2λ = cartesian2barycentric_setup(xs)
        λ = cartesian2barycentric(x2λ, x)
        # delta = S(0)
        delta = 10 * eps(S)
        if all(λi -> -delta <= λi <= 1 + delta, λ)
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
                # y += ys[n] * basis_x(Form{D,R}, x2λ, dλ2dx, n, x)
                y += map(b -> ys[n] * b, basis_x(Form{D,R}, x2λ, dλ2dx, n, x))
            end
            return y
        end
    end
    return error("Coordinate $x not found in manifold $(mfd.name)")
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
        x = mfd.coords[R][i]
        y = f(x)
        y::Form{D,R,T}
        values[i] = y[]::T
    end
    return Fun{D,P,R,C,S,T}(mfd, values)
end

################################################################################

@generated function integration_scheme(::Type{T}, ::Val{D},
                                       ::Val{order}) where {T,D,order}
    return grundmann_moeller(T, Val(D), order + iseven(order))
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
            x2λ = cartesian2barycentric_setup(xs)

            # `findfirst` here works only for R == 0
            @assert R == 0
            n = findfirst(==(i), sj)
            @assert n !== nothing
            # `[]` here works only for R == 0
            @assert R == 0
            kernel(x::SVector{C,T}) = basis_x(x2λ, n, x) * f(x)[]

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
                x2λ = cartesian2barycentric_setup(xs)

                # Find basis functions for simplices i and j
                @assert R == 0
                ni = findfirst(==(i), sk)
                nj = findfirst(==(j), sk)
                function kernel(x::SVector{C,S})
                    return basis_x(x2λ, ni, x) * basis_x(x2λ, nj, x)
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
    λ = cartesian2barycentric(setup, x)
    return basis_λ(n, λ)
end

# TODO: Remove this, use Bernstein polynomials instead
@fastmath function basis_λ(n::Int, λ::SVector{N,S}) where {N,S}
    @assert 1 <= n <= N
    return λ[n]
end

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

# Expressed in terms of barycentric coordinates:

# x = xⁱ λᵢ = x¹ λ₁ + x² λ₂ + x³ λ₃
# y = yⁱ λᵢ = y¹ λ₁ + y² λ₂ + y³ λ₃
# 1 = Σᵢ λᵢ = λ₁ + λ₂ + λ₃
# dx = xⁱ dλᵢ = x¹ dλ₁ + x² dλ₂ + x³ dλ₃
# dy = yⁱ dλᵢ = y¹ dλ₁ + y² dλ₂ + y³ dλ₃
# 0  = Σᵢ dλᵢ = dλ₁ + dλ₂ + dλ₃

# D     R     [n]
#
# 2     0     λ₁   λ₂   λ₃
# 2     1     (λ₁ + λ₂) (dλ₁ + dλ₂)   (λ₁ + λ₃) (dλ₁ + dλ₃)   (λ₂ + λ₃) (dλ₂ + dλ₃)
# 2     2     (λ₁ + λ₂ + λ₃) (dλ₁₂ + dλ₁₃ + dλ₂₃)
#
# 3     0     λ₁   λ₂   λ₃   λ₄
# 3     1     (λ₁ + λ₂) (dλ₁ + dλ₂)   (λ₁ + λ₃) (dλ₁ + dλ₃)   (λ₁ + λ₄) (dλ₁ + dλ₄)   (λ₂ + λ₃) (dλ₂ + dλ₃)   (λ₂ + λ₄) (dλ₂ + dλ₄)   (λ₃ + λ₄) (dλ₃ + dλ₄)
# 3     2     (λ₁ + λ₂ + λ₃) (dλ₁₂ + dλ₁₃ + dλ₂₃)   (λ₁ + λ₂ + λ₄) (dλ₁₂ + dλ₁₄ + dλ₂₄)   (λ₁ + λ₃ + λ₄) (dλ₁₃ + dλ₁₄ + dλ₃₄)   (λ₂ + λ₃ + λ₄) (dλ₂₃ + dλ₂₄ + dλ₃₄)
# 3     3     (λ₁ + λ₂ + λ₃ + λ₄) (dλ₁₂₃ + dλ₁₂₄ + dλ₁₃₄ + dλ₂₃₄)

# dx1 = x1i dλi
# dx1 ∧ dx2 = x1i dλi ∧ x2j dλj = x1i x2j dλi ∧ dλj
# dx1 ∧ dx2 ∧ dx3 = x1i x2j x3k dλi ∧ dλj ∧ dλk

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
    @assert N >= 1
    D = N - 1
    R::Int
    @assert 0 <= R <= D
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
    @assert 0 <= R <= D
    N::Int
    @assert N == D + 1
    C::Int
    @assert C == D
    # Number of barycentric form elements
    Nλ = length(Form{D + 1,R})
    # Number of physical form elements
    Nx = length(Form{D,R})

    r = ones(SMatrix{Nx,Nλ,T})
    for nx in 1:Nx, nλ in 1:Nλ
        lstx = DifferentialForms.Forms.lin2lst(Val(D), Val(R), nx)
        lstλ = DifferentialForms.Forms.lin2lst(Val(D + 1), Val(R), nλ)
        relt = one(T)
        for lx in lstx, lλ in lstλ
            relt *= xs[lλ][lx]
        end
        r = setindex(r, relt, nx, nλ)
    end
    return r
end

#TODO @fastmath
function basis_λ(::Type{<:Form{D,R}}, i::Int, λ::SVector{N,T}) where {D,R,C,N,T}
    D::Int
    R::Int
    @assert 0 <= R <= D
    # Number of barycentric coordinates
    @assert N == D + 1
    # Number of barycentric form elements
    Nλ = length(Form{D + 1,R})
    # Number of basis functions
    I = DifferentialForms.binomial(Val(D + 1), Val(R + 1))
    @assert 1 <= i <= I

    inds = DifferentialForms.Forms.lin2lst(Val(D + 1), Val(R + 1), i)
    rval = sum(λ[inds])

    bits = DifferentialForms.Forms.lin2bit(Val(D + 1), Val(R + 1), i)

    relts = zero(SVector{Nλ,T})
    for ind in inds
        bits2 = Base.setindex(bits, false, ind)
        j = DifferentialForms.Forms.bit2lin(Val(D + 1), Val(R), bits2)
        relts = setindex(relts, relts[j] + rval, j)
    end

    return Form{D + 1,R}(relts)
end

end
