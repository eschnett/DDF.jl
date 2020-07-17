using DDF

using ComputedFieldTypes
using StaticArrays
using Test

@testset "Geometry D=$D" for D in 1:Dmax
    T = Float64

    @testset "Empty topology" begin
        topo = Topology(Val(D))
        xs = Fun{D, Pr, 0}(topo, fulltype(Form{D, 1, T})[])
        geom = Geometry(topo.name, topo, xs)

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
        topo = Topology(Simplex(SVector{D + 1}(1:D+1)))
        xs = Fun{D, Pr, 0}(topo, regular_simplex(Form{D, D, T}))
        geom = Geometry(topo.name, topo, xs)

        for R in 0:D
            @test length(geom.volumes[R].values) == binomial(D + 1, R + 1)
        end
        # <https://en.wikipedia.org/wiki/Simplex#Volume>
        @test geom.volumes[D].values[1] ≈ sqrt(T(D + 1) / 2^D) / factorial(D)
    end



    # Cubes are not Delaunay
    # @testset "Cube" begin
    #     topo = hypercube_topology(Val(D))
    #     xs = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #         topo,
    #         [Chain{V,1}(SVector{D}(T((n-1) & (1<<(d-1)) != 0) for d in 1:D))
    #          for n in 1:1<<D])
    #     geom = Geometry(topo.name, topo, xs)
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
    geom0 = Geometry(topo0.name, topo0, xs0)

    # Not Delaunay
    # # Simplex
    # topo1 = Topology(Simplex(SVector{D+1}(1:D+1)))
    # xs1 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #     topo1,
    #     [Chain{V,1}(SVector{D}(T(d+1 == n) for d in 1:D)) for n in 1:D+1])
    # geom1 = Geometry("orthogonal simplex D=$D", topo1, dom, xs1)

    # Regular simplex
    topo2 = Topology(Simplex(SVector{D + 1}(1:(D+1))))
    xs2 = Fun{D, Pr, 0}(topo2, regular_simplex(Form{D, D, T}))
    geom2 = Geometry(topo2.name, topo2, xs2)

    # Not Delaunay
    # # Cube
    # topo3 = hypercube_topology(Val(D))
    # xs3 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #     topo3,
    #     [Chain{V,1}(SVector{D}(T((n-1) & (1<<(d-1)) != 0) for d in 1:D))
    #      for n in 1:1<<D])
    # geom3 = Geometry(topo3.name, topo3, dom, xs3)

    # geoms = [geom0, geom1, geom2, geom3]
    geoms = [geom0, geom2]

    for geom in geoms
        f0 = ones(Fun{D, P, R, T}, geom.topo)
        f1 = id(Fun{D, P, R, T}, geom.topo)
        fs = [f0, f1]
        if R == 0 && P == Pr
            for d in 1:D
                if geom.topo.nvertices == 0
                    # Cannot to deduce element type
                    xs = T[]
                else
                    xs = map(x -> x[d], geom.coords.values)
                end
                f = Fun{D, P, R}(geom.topo, xs)
                push!(fs, f)
            end
        end

        for f in fs
            h = hodge(Val(P), Val(R), geom)
            @test all(>(0), h.values.diag)
            hf = h * f
            hf::Fun{D, !P, R, T}
        end

        if P == Pr
            if R > 0
                c = coderiv(Val(P), Val(R), geom)
                for f in fs
                    cf = c * f
                    cf::Fun{D, P, R - 1, T}
                end
            end

            for f in fs
                l = laplace(Val(P), Val(R), geom)
                lf = l * f
                lf::Fun{D, P, R, T}
            end

        end
    end
end


@testset "Evaluate functions D=$D P=$P R=$R" for D in 1:Dmax,
    P in (Pr, Dl),
    R in 0:D

    # TODO: all R, all P
    (R == 0 && P == Pr) || continue

    T = Float64

    # Empty topology
    topo0 = Topology(Val(D))
    xs0 = Fun{D, Pr, 0}(topo0, fulltype(Form{D, 1, T})[])
    geom0 = Geometry(topo0.name, topo0, xs0)

    # Not Delaunay
    # # Simplex
    # topo1 = Topology(Simplex(SVector{D+1}(1:D+1)))
    # xs1 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #     topo1,
    #     [Chain{V,1}(SVector{D}(T(d+1 == n) for d in 1:D)) for n in 1:D+1])
    # geom1 = Geometry("orthogonal simplex D=$D", topo1, dom, xs1)

    # Regular simplex
    topo2 = Topology(Simplex(SVector{D + 1}(1:(D+1))))
    xs2 = Fun{D, Pr, 0}(topo2, regular_simplex(Form{D, D, T}))
    geom2 = Geometry(topo2.name, topo2, xs2)

    # Not Delaunay
    # # Cube
    # topo3 = hypercube_topology(Val(D))
    # xs3 = Fun{D, Pr, 0, fulltype(Chain{V,1,T})}(
    #     topo3,
    #     [Chain{V,1}(SVector{D}(T((n-1) & (1<<(d-1)) != 0) for d in 1:D))
    #      for n in 1:1<<D])
    # geom3 = Geometry(topo3.name, topo3, dom, xs3)

    # geoms = [geom0, geom1, geom2, geom3]
    geoms = [geom0, geom2]

    for geom in geoms
        f0 = ones(Fun{D, P, R, T}, geom.topo)
        f1 = id(Fun{D, P, R, T}, geom.topo)
        f2 = geom.coords
        fs = [f0, f1, f2]
        # for d in 1:D
        #     f = map(s->s[d], geom.coords.values)
        #     push!(f, fs)
        # end

        for f in fs

            # Interpolating at nodes must give nodal values
            for n in 1:geom.topo.nvertices
                x = geom.coords.values[n]
                @test evaluate(geom, f, x) == f.values[n]
            end

            # Interpolate at origin
            if geom.topo.nvertices > 0
                # TODO: all D
                if D == 1
                    x0 = Form{D, 1, T}(SVector(0))
                    if f == f0
                        @test evaluate(geom, f0, x0) == 1
                    elseif f == f1
                        # do nothing
                    elseif f == f2
                        @test evaluate(geom, f2, x0) == x0
                    end
                end
            end
        end
    end
end

# @testset "Derivative is functorial D=$D R=$R" for D in 1:Dmax, R in 0:D-1
#     T = Float64
# 
#     # Regular simplex
#     topo = Topology(Simplex(SVector{D+1}(1:D+1)))
#     xs = Fun{D, Pr, 0}(topo, regular_simplex(Form{D, D, T}))
#     geom = Geometry(topo, xs)
# 
#     for dir in 1:D, pow in 0:1
#         pow==0 && dir>1 && continue
#         @show D R dir pow
# 
#         g(x) = x[dir]^pow
#         dg(d,x) = d==dir ? (pow==0 ? zero(T) : pow*x[dir]^(pow-1)) : zero(T)
# 
#         xs = coordinates(Val(Pr), Val(0), geom)
#         @show xs.values
#         dxs = coordinates(Val(Pr), Val(1), geom)
#         @show dxs.values
#         f = map(g, xs)
#         @show f.values
#         d = deriv(Val(Pr), Val(0), geom.topo)
#         df = d*f
#         @show df.values
#         for i in 1:size(R+1, geom.topo)
#             # for 
#             @assert df.values[i] == dgdxs.values[i]
#         end
#     end
# end
