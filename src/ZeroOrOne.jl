module ZeroOrOne

export ZeroOne
"""
Zero or some (but we don't know how one)
"""
struct ZeroOne <: Number
    some::Bool
end

Base.:(==)(a::ZeroOne, b::ZeroOne) = a.some == b.some
Base.:(<)(a::ZeroOne, b::ZeroOne) = a.some < b.some
Base.hash(z::ZeroOne, h::UInt) = hash(0xcfe01124, hash(a.some, h))

Base.zero(::ZeroOne) = ZeroOne(false)
Base.one(::ZeroOne) = ZeroOne(true)
Base.Bool(a::ZeroOne) = a.some
Base.Int8(a::ZeroOne) = Int8(Bool(a))
Base.Int(a::ZeroOne) = Int(Bool(a))

# ZeroOne(b::Bool)
ZeroOne(b::Integer) = ZeroOne(b != 0)

Base.:+(a::ZeroOne) = a
Base.:+(a::ZeroOne, b::ZeroOne) = ZeroOne(a.some | b.some)
Base.:*(a::ZeroOne, b::ZeroOne) = ZeroOne(a.some & b.some)
Base.abs(a::ZeroOne) = a
Base.abs2(a::ZeroOne) = a

################################################################################

export One
"""
One (always has the same value)
"""
struct One <: Number end

Base.:(==)(::One, ::One) = true
Base.:(<)(::One, ::One) = false
Base.hash(::One, h::UInt) = hash(0xcfe01124, h)

Base.zero(::One) = zero(ZeroOne)
Base.one(::One) = One()
Base.Bool(::One) = true
Base.Int8(::One) = Int8(Bool(One()))
Base.Int(::One) = Int(Bool(One()))

One(b::Bool) = (@assert b; One())
One(b::Integer) = (@assert b == 1; One())

Base.:+(::One) = One()
Base.:+(::One, ::One) = One()
Base.:*(::One, ::One) = One()
Base.abs(::One) = One()
Base.abs2(::One) = One()

################################################################################

Base.promote_rule(::Type{ZeroOne}, ::Type{One}) = ZeroOne
Base.promote_rule(::Type{ZeroOne}, N::Type{<:Number}) = N
Base.promote_rule(::Type{One}, N::Type{<:Number}) = N

ZeroOne(::One) = ZeroOne(Bool(One()))
One(a::ZeroOne) = One(Bool(a))

end
