using DDF

using Test




@testset "Form D=$D R=$R" for D = 0:Dmax, R = 0:D
    # Using === instead of == for comparisons to catch wrong types
    T = Rational{Int64}
    n = zero(Form{D,R,T})
    x = rand(Form{D,R,T})
    y = rand(Form{D,R,T})
    z = rand(Form{D,R,T})
    a = rand(T)
    b = rand(T)

    # Vector space

    @test +x === x
    @test (x + y) + z === x + (y + z)

    @test n + x === x
    @test x + n === x

    @test x + y === y + x

    @test x + (-x) === n
    @test (-x) + x === n
    @test x - y === x + (-y)

    @test (a * b) * x === a * (b * x)
    @test x * (a * b) === (x * a) * b

    @test one(T) * a === a
    @test one(T) * x === x
    @test x * one(T) === x

    @test zero(T) * a === zero(T)
    @test zero(T) * x === n
    @test x * zero(T) === n

    if a != zero(T)
        @test a * inv(a) === one(T)

        @test inv(a) * (a * x) === x
        @test (x * a) * inv(a) === x

        @test inv(a) * x === a \ x
        @test x * inv(a) === x / a
    end

    @test (a + b) * x === a * x + b * x
    @test x * (a + b) === x * a + x * b

    @test a * (x + y) === a * x + a * y
    @test (x + y) * a === x * a + y * a

    # Nonlinear transformations
    @test map(+, x) === x
    @test map(+, x, y) === x + y
    @test map(+, x, y, z) === x + y + z
end



@testset "Form D=$D R1=$R1 R2=$R2 R3=$R3" for D = 1:Dmax,
    R1 = 0:D,
    R2 = 0:D-R1,
    R3 = 0:D-R1-R2
    # Using === instead of == for comparisons to catch wrong types
    T = Rational{Int64}
    e = one(Form{D,0,T})
    x = rand(Form{D,R1,T})
    y = rand(Form{D,R2,T})
    y2 = rand(Form{D,R2,T})
    z = rand(Form{D,R3,T})
    a = rand(T)
    b = rand(T)

    # Multiplicative structure

    # various duals
    @test ~~x === x
    @test conj(conj(x)) === x
    @test inv(inv(x)) === x      # can fail
    @test transpose(transpose(x)) === x
    @test ⋆⋆x === bitsign(R1 * (D - R1)) * x
    @test inv(⋆)(⋆x) === x
    @test ⋆inv(⋆)(x) === x
    @test ⋆⋆ ⋆ ⋆x === x

    # exterior product: x ∧ y
    (x ∧ y)::Form{D,R1 + R2}

    @test (x ∧ y) ∧ z === x ∧ (y ∧ z)

    @test e ∧ x === x
    @test x ∧ e === x

    @test x ∧ zero(y) === zero(x ∧ y)
    @test zero(y) ∧ x === zero(y ∧ x)

    @test a * (x ∧ y) === x ∧ (a * y)
    @test x ∧ (y + y2) === x ∧ y + x ∧ y2

    if R1 == 0 || R2 == 0
        @test x ∧ y === y ∧ x
    elseif R1 == R2 == 1
        @test x ∧ y === -(y ∧ x)
    else
        # do nothing
    end

    # regressive product: ⋆(x ∨ y) = ⋆x ∧ ⋆y
    @test ⋆(⋆x ∨ ⋆y) === ⋆⋆x ∧ ⋆ ⋆ y

    # dot product: x ⋅ y = x ∨ ⋆y   (right contraction)
    @test ⋆(⋆x ⋅ y) === ⋆⋆x ∧ ⋆ ⋆ y

    # cross product: x × y = ⋆(x ∧ y)
    @test x × y === ⋆(x ∧ y)
end



# @testset "Homogenous Geometric Algebra D=$D" for D in 1:Dmax
#     S = Signature(D)
#     V = SubManifold(S)
#     @test iseuclidean(V)
#     T = Rational{Int128}
#     for n in 1:100
#         x = rand(Chain{V,1,T})
#         x::Chain
#         @test iseuclidean(x)
#         px = homogenous(x)
#         px::Chain
#         @test ishomogenous(px)
#         @test euclidean(px) == x
#         # TODO: add more checks
#     end
# end
# 
# 
# 
# @testset "Conformal Geometric Algebra D=$D" for D in 1:Dmax
#     S = Signature(D)
#     V = SubManifold(S)
#     @test iseuclidean(V)
#     T = Rational{Int128}
#     for n in 1:100
#         x = rand(Chain{V,1,T})
#         x::Chain
#         @test iseuclidean(x)
#         cx = conformal(x)
#         cx::Chain
#         @test isconformal(cx)
#         @test euclidean(cx) == x
#         # TODO: add more checks
#     end
# end
