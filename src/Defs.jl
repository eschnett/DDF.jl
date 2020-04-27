using StaticArrays



export sarray
@generated function sarray(::Type{T}, f::F, ::Val{R}) where {T, F, R}
    R::Integer
    quote
        SArray{Tuple{$R}, T}($([:(f($i)) for i in 1:R]...))
    end
end
@generated function sarray(::Type{T}, f::F, ::Val{R1}, ::Val{R2}
                           ) where {T, F, R1, R2}
    R1::Integer
    R2::Integer
    quote
        SArray{Tuple{$R1, $R2}, T}(
            $([:(f($i, $j)) for i in 1:R1, j in 1:R2]...))
    end
end

@generated function Base.sum(::Type{T}, f::F, ::Val{R}) where {T, F, R}
    R::Integer
    if R <= 0
        quote
            zero(T)
        end
    else
        quote
            (+($([:(T(f($i))) for i in 1:R]...)))::T
        end
    end
end
@generated function Base.sum(::Type{T}, f::F, ::Val{R1}, ::Val{R2}
                             ) where {T, F, R1, R2}
    R1::Integer
    R2::Integer
    if R1 <= 0 || R2 <= 0
        quote
            zero(T)
        end
    else
        quote
            (+($([:(T(f($i, $j))) for i in 1:R1, j in 1::R2]...)))::T
        end
    end
end
