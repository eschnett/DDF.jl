using DDF

using BitIntegers
using StaticArrays
using Test



@testset "Circumcentre D=$D N=$N" for D in 1:Dmax, N in 1:D+1
    T = Rational{Int256}
    for N in 1:D+1
        xs = SVector{N}(rand(Form{D,1,T}) for n in 1:N)
        cc = circumcentre(xs)
        cc::Form{D,1,T}
        # Check that circumcentre has same distance from all vertices
        r2 = abs2(xs[1] - cc)
        @test all(==(r2), abs2(xs[n] - cc) for n in 1:N)
        # Check that circumcentre lies in hyperplane spanned by vertices
        subspace = ∧(xs...)
        @test iszero(cc ∧ subspace)
    end
end



@testset "Regular simplex D=$D" for D in 1:Dmax
    N = D+1
    T = Float64
    s = regular_simplex(Form{D, D, T})
    for i in 1:N, j in i+1:N
        @test abs2(s[i] - s[j]) ≈ 1
    end
end
