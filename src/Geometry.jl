using ComputedFieldTypes
using StaticArrays

export Domain
struct Domain{D, T}
    xmin::NTuple{D, T}
    xmax::NTuple{D, T}
end



export Coords
struct Coords{D, T}
    mf::Manifold{D}
    dom::Domain{D}
    coords::NTuple{D, Fun{D, 0, T}}
    # TODO: Store hodge star here
end



export sarray
@generated function sarray(::Type{T}, f::F, ::Val{R}) where {T, F, R}
    quote
        SArray{Tuple{$(length(R))}, $T}($([:(f($i)) for i in R]...))
    end
end
@generated function sarray(::Type{T}, f::F, ::Val{R1}, ::Val{R2}
                           ) where {T, F, R1, R2}
    quote
        SArray{Tuple{$(length(R1)), $(length(R2))}, $T}(
            $([:(f($i, $j)) for i in R1, j in R2]...))
    end
end

function Base.sum(::Type{T}, f::F, range::R) where {T, F, R}
    s = T(0)
    for i in range
        s += f(i)::T
    end
    s
end
function Base.sum(::Type{T}, f::F, range1::R1, range2::R2) where {T, F, R1, R2}
    s = T(0)
    for i in range1, j in range2
        s += f(i, j)::T
    end
    s
end

export circumcentre
function circumcentre(xs::SVector{R, SVector{D, T}}
                      )::SVector{D, T} where {R, D, T}
    # G. Westendorp, A formula for the N-circumsphere of an N-simplex,
    # <https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm>,
    # April 2013.
    CM = sarray(T,
                (i, j) -> (
                    (i == 1 || j == 1) ?
                    i != j :
                    sum(T, c -> (xs[i-1][c] - xs[j-1][c])^2, 1:D)),
                Val(1:R+1), Val(1:R+1))
    CM1 = inv(CM)
    r2 = CM1[1,1] / -2
    α = sarray(T, i -> CM1[i+1,1], Val(1:R))
    cc = sarray(T, b -> sum(T, j -> α[j] * xs[j][b], 1:R), Val(1:D))
    # Check radii
    for j in 1:R
        rj2 = sum(T, b -> (xs[j][b] - cc[b])^2, 1:D)
        @assert abs(r2 - rj2) <= T(1.0e-12) * r2
    end
    cc
end



export circumcentres
function circumcentres(coords::Coords{D, T})::Vector{NTuple{D, T}} where {D, T}
    mf = coords.mf
    n = dim(Val(D), mf)
    ccs = Array{NTuple{D, T}}(undef, n)
    if D == 0
        return ccs
    end
    simplices = mf.simplices[D]
    for (i,s) in enumerate(simplices)
        # cs[a][b] = coords of vertex a of this simplex
        cs = ntuple(j -> ntuple(b -> coords.coords[b][s[j]], D), D+1)
        # G. Westendorp, A formula for the N-circumsphere of an
        # N-simplex,
        # <https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm>,
        # April 2013.
        CM = Array{T}(undef, D+2, D+2)
        for a in 1:D+2, b in 1:D+2
            if a == 1 || b == 1
                CM[a,b] = a != b
            else
                CM[a,b] = sum((cs[a-1][c] - cs[b-1][c])^2 for c in 1:D)
            end
        end
        CM1 = inv(CM)
        r2 = CM1[1,1] / -2
        α = T[CM1[a+1,1] for a in 1:D+1]
        cc = ntuple(b -> sum(α[j] * cs[j][b] for j in 1:D+1), D)
        # Check radii
        # TODO: Turn this into a test case
        for j in 1:D+1
            rj2 = sum((cs[j][b] - cc[b])^2 for b in 1:D)
            @assert abs(r2 - rj2) <= 1.0e-12 * r2
        end
        # TODO: Check Delauney condition
        ccs[i] = cc
    end
    ccs
end



export hodge
function hodge(coords::Coords{D, T}
               )::NTuple{max(0, D-1), Vector{T}} where {D, T}
    mf = coords.mf

    # ccs = circumcentres(coords)

    dualVs = Array{Vector{T}}(undef, max(0, D-1))

    for R in D-1:-1:1
        R1 = R + 1

        dualV1 = R1 == D ? nothing : dualVs[R1]
        dualV = Array{T}(undef, dim(Val(R), mf))

        # TODO: use boundary operator to find connectivity
        for (i,si) in enumerate(mf.simplices[R])
            @assert length(si) == R+1
            xis = sarray(fulltype(SVector{D,T}),
                         k -> sarray(T, a -> coords.coords[a][si[k]], Val(1:D)),
                         Val(1:R+1))
            cci = circumcentre(xis)
            V = T(0)
            for (j,sj) in enumerate(mf.simplices[R1])
                if any(sj[k] ==i for k in 1:length(sj))
                    b = R1 == D ? T(1) : dualV1[j]
                    @assert length(sj) == R1+1
                    xjs = sarray(
                        fulltype(SVector{D,T}),
                        k -> sarray(T, a -> coords.coords[a][sj[k]], Val(1:D)),
                        Val(1:R1+1))
                    ccj = circumcentre(xjs)
                    h = sqrt(sum(T, a -> (cci[a] - ccj[a])^2, 1:D))
                    V += b * h / (D - R)
                end
            end
            dualV[i] = V
        end

        dualVs[R] = dualV
    end

    tuple(dualVs...)
end



# @computed struct Tree{D, T}
#     dom::Domain{D}
#     mf::Manifold{D}
# 
#     pivot::NTuple{D, T}
#     tree::Union{NTuple{2^D, fulltype(Tree{D, T})},
#                 Vector{Int}}
# end



# function sample
# function evaluate
