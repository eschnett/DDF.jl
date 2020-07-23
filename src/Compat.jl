module Compat

using Grassmann



# Ensure type stability in some methods
# <https://github.com/chakravala/Grassmann.jl/issues/65>
function Grassmann.:↑(ω::T) where {T<:TensorAlgebra}
    V = Manifold(ω)
    !(hasinf(V) || hasorigin(V)) && (return ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        ((G.v∞ * (one(valuetype(ω)) / 2)) * ω^2 + G.v∅) + ω
    else
        ω2 = ω^2
        iω2 = inv(ω2 + 1)
        (hasinf(V) ? G.v∞ : G.v∅) * (ω2 - 1) * iω2 + 2 * iω2 * ω
    end
end
function Grassmann.:↓(ω::T) where {T<:TensorAlgebra}
    V = Manifold(ω)
    !(hasinf(V) || hasorigin(V)) && (return ω)
    G = Λ(V)
    return if hasinf(V) && hasorigin(V)
        inv(one(valuetype(ω)) * G.v∞∅) * (G.v∞∅ ∧ ω) / (-ω ⋅ G.v∞)
    else
        b = hasinf(V) ? G.v∞ : G.v∅
        ((ω ∧ b) * b) / (1 - b ⋅ ω)
    end
end

# Override (aka delete) harmful methods from Grassmann
# <https://github.com/chakravala/Grassmann.jl/issues/67>
Base.ndims(::Vector{Chain{V,G,T,X}} where {G,T,X}) where {V} = 1
# Base.parent(::Vector{Chain{V,G,T,X}} where {G,T,X}) where V = ???

# Implement a unary version of ∧
# <https://github.com/chakravala/Grassmann.jl/issues/69>
Grassmann.:∧(x::Chain) = x
Grassmann.:∧(x::Simplex) = x
Grassmann.:∧(x::MultiVector) = x

end
