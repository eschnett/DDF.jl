using DDF

using ComputedFieldTypes
using StaticArrays
using Test



@testset "Geometry D=$D" for D in 1:Dmax
    T = Float64

    @testset "Empty topology" begin
        topo = Topology(Val(D))
        xs = Fun{D, Pr, 0}(topo, fulltype(Form{D, 1, T})[])
        geom = Geometry(topo, xs)

        for R in 0:D
            @test length(geom.volumes[R].values) == 0
        end
    end



    # Orthogonal simplices are not Delaunay
    # @testset "Orthogonal simplex" begin
    #     topo = Topology(Simplex(SVector{D+1}(1:D+1)))
    #     xs = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #         topo,
    #         [Chain{V,1}(SVector{D}(T(d+1 == n) for d in 1:D)) for n in 1:D+1])
    #     geom = Geometry(topo, xs)
    # 
    #     for R in 0:D
    #         @test length(geom.volumes[R].values) == binomial(D+1, R+1)
    #     end
    #     @test geom.volumes[D].values[1] ≈ one(T)/factorial(D)
    # 
    #     # if D==2 || D==3
    #     #     @show geom.topo.nvertices
    #     #     @show geom.topo.simplices
    #     #     for d in 1:D
    #     #         @show d geom.topo.boundaries[d]
    #     #     end
    #     #     @show geom.coords.values
    #     #     for d in 1:D
    #     #         @show d geom.volumes[d].values
    #     #     end
    #     #     @show geom.dualcoords.values
    #     # end
    # end


    @testset "Regular simplex" begin
        topo = Topology(Simplex(SVector{D+1}(1:D+1)))
        xs = Fun{D, Pr, 0}(topo, regular_simplex(Form{D, D, T}))
        geom = Geometry(topo, xs)

        for R in 0:D
            @test length(geom.volumes[R].values) == binomial(D+1, R+1)
        end
        # <https://en.wikipedia.org/wiki/Simplex#Volume>
        @test geom.volumes[D].values[1] ≈ sqrt(T(D+1)/2^D) / factorial(D)
    end



    # Cubes are not Delaunay
    # @testset "Cube" begin
    #     topo = hypercube_topology(Val(D))
    #     xs = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #         topo,
    #         [Chain{V,1}(SVector{D}(T((n-1) & (1<<(d-1)) != 0) for d in 1:D))
    #          for n in 1:1<<D])
    #     geom = Geometry(topo, xs)
    # 
    #     @test length(geom.volumes[0].values) == 2^D
    #     @test length(geom.volumes[D].values) == factorial(D)
    #     @test sum(geom.volumes[D].values) ≈ 1
    # end
end



@testset "Geometry ops D=$D P=$P R=$R" for D in 1:Dmax, P in (Pr, Dl), R in 0:D
    T = Float64

    # Empty topology
    topo0 = Topology(Val(D))
    xs0 = Fun{D, Pr, 0}(topo0, fulltype(Form{D, 1, T})[])
    geom0 = Geometry(topo0, xs0)

    # Not Delaunay
    # # Simplex
    # topo1 = Topology(Simplex(SVector{D+1}(1:D+1)))
    # xs1 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #     topo1,
    #     [Chain{V,1}(SVector{D}(T(d+1 == n) for d in 1:D)) for n in 1:D+1])
    # geom1 = Geometry(topo1, dom, xs1)

    # Regular simplex
    topo2 = Topology(Simplex(SVector{D+1}(1:D+1)))
    xs2 = Fun{D, Pr, 0}(topo2, regular_simplex(Form{D, D, T}))
    geom2 = Geometry(topo2, xs2)

    # Not Delaunay
    # # Cube
    # topo3 = hypercube_topology(Val(D))
    # xs3 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #     topo3,
    #     [Chain{V,1}(SVector{D}(T((n-1) & (1<<(d-1)) != 0) for d in 1:D))
    #      for n in 1:1<<D])
    # geom3 = Geometry(topo3, dom, xs3)

    # geoms = [geom0, geom1, geom2, geom3]
    geoms = [geom0, geom2]

    for geom in geoms

        f0 = ones(Fun{D, P, R, T}, geom.topo)
        f1 = id(Fun{D, P, R, T}, geom.topo)
        fs = [f0, f1]

        for f in fs
            h = hodge(Val(P), Val(R), geom)
            @test all(>(0), h.values.diag)
            hf = h*f
            hf::Fun{D, !P, R, T}
        end

        if P == Pr
            if R > 0
                c = coderiv(Val(P), Val(R), geom)
                for f in fs
                    cf = c*f
                    cf::Fun{D, P, R-1, T}
                end
            end

            for f in fs
                l = laplace(Val(P), Val(R), geom)
                lf = l*f
                lf::Fun{D, P, R, T}
            end
        end
    end
end



# @testset "Derivative is functorial D=$D R=$R" for D in 1:Dmax, R in 0:D-1
#     S = Signature(D)
#     V = SubTopology(S)
#     T = Float64
# 
#     dom = Domain(Chain{V,1}(zeros(SVector{D,T})),
#                  Chain{V,1}(ones(SVector{D,T})))
#     # Regular simplex
#     topo = Topology(Simplex(SVector{D+1}(1:D+1)))
#     xs = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(topo, regular_simplex(Val(D), T))
#     geom = Geometry(topo, dom, xs)
# 
#     for dir in 1:D, pow in 0:1
#         pow==0 && dir>1 && continue
#         @show D R dir pow
# 
#         g(x) = x[dir]^pow
#         dg(d,x) = d==dir ? (pow==0 ? zero(T) : pow*x[dir]^(pow-1)) : zero(T)
# 
#         xs = coords(Val(0), Val(Pr), geom)
#         @show xs.values
#         dxs = coords(Val(1), Val(Pr), geom)
#         @show dxs.values
#         f = map(g, xs)
#         @show f.values
#         d = deriv(Val(Pr), Val(0), geom.topo)
#         df = d*f
#         @show df.values
#         for i in 1:size(R+1, geom.topo)
#             for 
#             @assert df.values[i] == dgdxs.values[i]
#         end
#     end
# end
