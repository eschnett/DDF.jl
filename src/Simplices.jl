module Simplices

using StaticArrays

using ..Defs

export Simplex
"""
Oriented simplex
"""
struct Simplex{N,T}
    vertices::SVector{N,T}
    signbit::Bool

    function Simplex{N,T}(vertices::SVector{N,T},
                          signbit::Bool = false) where {N,T}
        N::Int
        T::Type
        v, s = sort_perm(vertices)
        return new{N,T}(v, signbit ⊻ isodd(s))
    end
    function Simplex(vertices::SVector{N,T}, signbit::Bool = false) where {N,T}
        return Simplex{N,T}(vertices, signbit)
    end
end

function Defs.invariant(s::Simplex)::Bool
    return issorted(s.vertices)
end

function Base.show(io::IO, s::Simplex)
    sign = s.signbit ? "-" : "+"
    return print(io, "⟨$sign⟩$(s.vertices)")
end

function Base.:(==)(s::S, t::S) where {S<:Simplex}
    return s.vertices == t.vertices && s.signbit == t.signbit
end
function Base.isless(s::S, t::S) where {S<:Simplex}
    isless(s.vertices, t.vertices) && return true
    isless(t.vertices, s.vertices) && return false
    return isless(s.signbit, t.signbit)
end

Base.ndims(::Type{S}) where {S<:Simplex} = length(S) - 1
Base.ndims(::S) where {S<:Simplex} = ndims(S)

Base.getindex(s::Simplex, i) = s.vertices[i]
Base.length(::Type{<:Simplex{N}}) where {N} = N
Base.length(::S) where {S<:Simplex} = length(S)

end
