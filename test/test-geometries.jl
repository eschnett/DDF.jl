using DDF

using ComputedFieldTypes
using StaticArrays
using Test

@testset "Geometry D=$D" for D = 1:Dmax
    T = Float64

    @testset "Empty topology" begin
        topo = Topology(Val(D))
        xs = Fun{D,Pr,0}(topo, fulltype(Form{D,1,T})[])
        geom = Geometry(topo.name, topo, xs)

        for R = 0:D
            @test length(geom.volumes[R].values) == 0
        end
    end



    @testset "Orthogonal simplex" begin
        topo = Topology(Simplex(SVector{D + 1}(1:D+1)))
        xs = Fun{D,Pr,0}(
            topo,
            [Form{D,1}(SVector{D}(T(d + 1 == n) for d = 1:D)) for n = 1:D+1],
        )
        geom = Geometry(topo.name, topo, xs)

        for R = 0:D
            @test length(geom.volumes[R].values) == binomial(D + 1, R + 1)
        end
        @test geom.volumes[D].values[1] ≈ one(T) / factorial(D)
    end



    @testset "Regular simplex" begin
        topo = Topology(Simplex(SVector{D + 1}(1:D+1)))
        xs = Fun{D,Pr,0}(topo, regular_simplex(Form{D,D,T}))
        geom = Geometry(topo.name, topo, xs)

        for R = 0:D
            @test length(geom.volumes[R].values) == binomial(D + 1, R + 1)
        end
        # <https://en.wikipedia.org/wiki/Simplex#Volume>
        @test geom.volumes[D].values[1] ≈ sqrt(T(D + 1) / 2^D) / factorial(D)
    end



    @testset "Cube" begin
        topo = hypercube_manifold(Val(D))
        xs = Fun{D,Pr,0}(
            topo,
            [
                Form{D,1}(SVector{D}(T((n - 1) & (1 << (d - 1)) != 0) for d = 1:D))
                for n = 1:1<<D
            ],
        )
        geom = Geometry(topo.name, topo, xs)

        @test length(geom.volumes[0].values) == 2^D
        @test length(geom.volumes[D].values) == factorial(D)
        @test sum(geom.volumes[D].values) ≈ 1
    end
end



@testset "Geometry ops D=$D P=$P R=$R" for D = 1:Dmax, P in (Pr, Dl), R = 0:D
    T = Float64

    # Empty topology
    topo0 = Topology(Val(D))
    xs0 = Fun{D,Pr,0}(topo0, fulltype(Form{D,1,T})[])
    geom0 = Geometry(topo0.name, topo0, xs0)

    # Orthogonal Simplex
    topo1 = Topology(Simplex(SVector{D + 1}(1:D+1)))
    xs1 =
        Fun{D,Pr,0}(topo1, [Form{D,1}(SVector{D}(T(d + 1 == n) for d = 1:D)) for n = 1:D+1])
    geom1 = Geometry("orthogonal simplex D=$D", topo1, xs1)

    # Regular simplex
    topo2 = Topology(Simplex(SVector{D + 1}(1:(D+1))))
    xs2 = Fun{D,Pr,0}(topo2, regular_simplex(Form{D,D,T}))
    geom2 = Geometry(topo2.name, topo2, xs2)

    # Cube
    topo3 = hypercube_manifold(Val(D))
    xs3 = Fun{D,Pr,0}(
        topo3,
        [
            Form{D,1}(SVector{D}(T((n - 1) & (1 << (d - 1)) != 0) for d = 1:D))
            for n = 1:1<<D
        ],
    )
    geom3 = Geometry(topo3.name, topo3, xs3)

    geoms = [geom0, geom1, geom2, geom3]

    for geom in geoms
        f0 = ones(Fun{D,P,R,T}, geom.topo)
        f1 = id(Fun{D,P,R,T}, geom.topo)
        fs = [f0, f1]
        if R == 0 && P == Pr
            for d = 1:D
                if geom.topo.nvertices == 0
                    # Cannot to deduce element type
                    xs = T[]
                else
                    xs = map(x -> x[d], geom.coords.values)
                end
                f = Fun{D,P,R}(geom.topo, xs)
                push!(fs, f)
            end
        end

        for f in fs
            h = hodge(Val(P), Val(R), geom)
            # Require Hodge operator to be at least positive semidefinite
            @test all(>=(0), h.values.diag)
            if geom ∉ [geom1, geom3]
                # We want it to be positive definite, but that's not
                # true (e.g. for orthogonal corners )
                @test all(>(0), h.values.diag)
            end
            hf = h * f
            hf::Fun{D,!P,R,T}
        end

        if P == Pr
            if R > 0
                c = coderiv(Val(P), Val(R), geom)
                for f in fs
                    cf = c * f
                    cf::Fun{D,P,R - 1,T}
                end
            end

            for f in fs
                l = laplace(Val(P), Val(R), geom)
                lf = l * f
                lf::Fun{D,P,R,T}
            end

        end
    end
end



@testset "Evaluate functions D=$D P=$P R=$R" for D = 1:Dmax, P in (Pr, Dl), R = 0:D

    # TODO: all R, all P
    (R == 0 && P == Pr) || continue

    T = Float64

    # Empty topology
    topo0 = Topology(Val(D))
    xs0 = Fun{D,Pr,0}(topo0, fulltype(Form{D,1,T})[])
    geom0 = Geometry(topo0.name, topo0, xs0)

    # Orthogonal Simplex
    topo1 = Topology(Simplex(SVector{D + 1}(1:D+1)))
    xs1 =
        Fun{D,Pr,0}(topo1, [Form{D,1}(SVector{D}(T(d + 1 == n) for d = 1:D)) for n = 1:D+1])
    geom1 = Geometry("orthogonal simplex D=$D", topo1, xs1)

    # Regular simplex
    topo2 = Topology(Simplex(SVector{D + 1}(1:(D+1))))
    xs2 = Fun{D,Pr,0}(topo2, regular_simplex(Form{D,D,T}))
    geom2 = Geometry(topo2.name, topo2, xs2)

    # Cube
    topo3 = hypercube_manifold(Val(D))
    xs3 = Fun{D,Pr,0}(
        topo3,
        [
            Form{D,1}(SVector{D}(T((n - 1) & (1 << (d - 1)) != 0) for d = 1:D))
            for n = 1:1<<D
        ],
    )
    geom3 = Geometry(topo3.name, topo3, xs3)

    geoms = [geom0, geom1, geom2, geom3]

    for geom in geoms
        f0 = zero(Fun{D,P,R,T}, geom.topo)
        f1 = ones(Fun{D,P,R,T}, geom.topo)
        f2 = id(Fun{D,P,R,T}, geom.topo)
        f3 = geom.coords
        fs = [f0, f1, f2, f3]

        for f in fs

            # Interpolating at nodes must give nodal values
            for n = 1:geom.topo.nvertices
                x = geom.coords.values[n]
                @test evaluate(geom, f, x) == f.values[n]
            end

            # Interpolate at origin
            if geom.topo.nvertices > 0
                # TODO: all D
                if D == 1
                    x0 = Form{D,1,T}(SVector(0))
                    if f == f0
                        @test evaluate(geom, f, x0) == 0
                    elseif f == f1
                        @test evaluate(geom, f, x0) == 1
                    elseif f == f2
                        # do nothing
                    elseif f == f3
                        @test evaluate(geom, f, x0) == x0
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
