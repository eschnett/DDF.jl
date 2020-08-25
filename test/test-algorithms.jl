using DDF

using DifferentialForms
using StaticArrays
using Test

@testset "Circumcentre D=$D N=$N" for D in 1:Dmax, N in 1:(D + 1)
    T = Rational{BigInt}
    xs = SVector{N}(rand(Form{D,1,T}) for n in 1:N)
    cc = circumcentre(xs)
    cc::Form{D,1,T}
    # Check that circumcentre has same distance from all vertices
    r2 = norm2(xs[1] - cc)
    @test all(==(r2), norm2(xs[n] - cc) for n in 1:N)
    # Check that circumcentre lies in hyperplane spanned by vertices
    if sum(frank.(xs)) > D
        @test true
    else
        subspace = ∧(xs...)
        if frank(cc) + frank(subspace) > D
            @test true
        else
            @test iszero(cc ∧ subspace)
        end
    end
end

#TODO @testset "Volume" begin end

#TODO @testset "Dual volume" begin end
