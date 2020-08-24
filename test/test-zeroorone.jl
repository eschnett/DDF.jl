using DDF

@testset "ZeroOne" begin
    @test ZeroOne(0) + ZeroOne(0) == ZeroOne(0)
    @test ZeroOne(0) + ZeroOne(1) == ZeroOne(1)
    @test ZeroOne(1) + ZeroOne(0) == ZeroOne(1)
    @test ZeroOne(1) + ZeroOne(1) == ZeroOne(1)
end

@testset "One" begin
    @test One(1) + One(1) == One(1)
    @test One(1) + ZeroOne(0) == ZeroOne(1)
    @test ZeroOne(0) + One(1) == ZeroOne(1)
    @test One(One(1) + ZeroOne(0)) == One(1)
    @test One(ZeroOne(0) + One(1)) == One(1)
end
