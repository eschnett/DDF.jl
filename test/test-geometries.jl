using DDF

using ComputedFieldTypes
using Random
using StaticArrays
using Test

const geometries = Dict{Int,Dict{String,Geometry}}()

@testset "Delaunay triangulation D=$D" for D in 1:Dmax
    T = Float64

    geometries[D] = Dict{String,Geometry}()

    @testset "Empty topology" begin
        topo = Topology(Val(D))
        xs = Fun{D,Pr,0}(topo, fulltype(Form{D,1,T})[])
        geom = Geometry(topo.name, topo, xs)
        geometries[D]["empty topology"] = geom

        @test isempty(geom.volumes[D].values)
    end

    @testset "Orthogonal simplex" begin
        coords = [Form{D,1}(SVector{D}(T(d + 1 == n) for d in 1:D))
                  for n in 1:(D + 1)]
        geom = delaunay("Orthogonal simplex", coords)
        geometries[D]["orthogonal simplex"] = geom

        vol = sum(geom.volumes[D].values)
        @test vol ≈ T(1) / factorial(D)
    end

    @testset "Regular simplex" begin
        coords = regular_simplex(Form{D,D,T})
        geom = delaunay("Regular simplex", coords)
        geometries[D]["regular simplex"] = geom

        vol = sum(geom.volumes[D].values)
        # <https://en.wikipedia.org/wiki/Simplex#Volume>
        @test vol ≈ sqrt(T(D + 1) / 2^D) / factorial(D)
    end

    @testset "Standard Hypercube" begin
        topo = hypercube_manifold(Val(D))
        xs = Fun{D,Pr,0}(topo,
                         [Form{D,1}(SVector{D}(T((n - 1) & (1 << (d - 1)) != 0)
                                               for d in 1:D))
                          for n in 1:(1 << D)])
        geom = Geometry(topo.name, topo, xs)
        geometries[D]["standard hypercube"] = geom

        vol = sum(geom.volumes[D].values)
        @test vol ≈ 1
    end

    @testset "Delaunay Hypercube" begin
        coords = fulltype(Form{D,1,T})[]
        imin = CartesianIndex(ntuple(d -> 0, D))
        imax = CartesianIndex(ntuple(d -> 1, D))
        for i in imin:imax
            push!(coords, Form{D,1,T}(SVector{D,T}(i.I...)))
        end
        push!(coords, Form{D,1,T}(SVector{D,T}(T(1) / 2 for d in 1:D)))
        geom = delaunay("Delaunay Hypercube", coords)
        geometries[D]["delaunay hypercube"] = geom

        vol = sum(geom.volumes[D].values)
        @test vol ≈ 1
    end

    @testset "Random Delaunay Hypercube" begin
        geom = missing
        while geom === missing
            coords = fulltype(Form{D,1,T})[]
            imin = CartesianIndex(ntuple(d -> 0, D))
            imax = CartesianIndex(ntuple(d -> 1, D))
            for i in imin:imax
                push!(coords, Form{D,1,T}(SVector{D,T}(i.I...)))
            end
            for R in 1:D
                for i in 1:(2 * D)
                    # coord = zero(SVector{D,T})
                    coord = zero(Form{D,1,T})
                    while true
                        # Choose a random corner
                        coord1 = SVector{D,T}(rand(0:1, D))
                        # Choose random coordinates for R directions
                        for d in randperm(D)[1:R]
                            # coord1 = setindex(coord1, rand(T), d)
                            coord1 = setindex(coord1, rand(1:9) / T(10), d)
                        end
                        coord = Form{D,1,T}(coord1)
                        all(!=(coord), coords) && break
                    end
                    push!(coords, coord)
                end
            end
            try
                geom = delaunay("Random Delaunay Hypercube", coords)
            catch ex
                ex isa ZeroVolumeException || rethrow(ex)
                @show ex
                println("Found zero-volume $(ex.R)-simplex; retrying...")
            end
        end

        geometries[D]["random delaunay hypercube"] = geom

        vol = sum(geom.volumes[D].values)
        @test vol ≈ 1
    end
end

@testset "Geometry D=$D" for D in 1:Dmax
    T = Float64

    @testset "Empty topology" begin
        geom = geometries[D]["empty topology"]

        for R in 0:D
            @test length(geom.volumes[R].values) == 0
        end
    end

    @testset "Orthogonal simplex" begin
        geom = geometries[D]["orthogonal simplex"]

        for R in 0:D
            @test length(geom.volumes[R].values) == binomial(D + 1, R + 1)
        end
        @test geom.volumes[D].values[1] ≈ one(T) / factorial(D)
    end

    @testset "Regular simplex" begin
        geom = geometries[D]["regular simplex"]

        for R in 0:D
            @test length(geom.volumes[R].values) == binomial(D + 1, R + 1)
        end
        # <https://en.wikipedia.org/wiki/Simplex#Volume>
        @test geom.volumes[D].values[1] ≈ sqrt(T(D + 1) / 2^D) / factorial(D)
    end

    @testset "Standard Hypercube" begin
        geom = geometries[D]["standard hypercube"]

        @test length(geom.volumes[0].values) == 2^D
        @test length(geom.volumes[D].values) == factorial(D)
        @test sum(geom.volumes[D].values) ≈ 1
    end

    @testset "Delaunay Hypercube" begin
        geom = geometries[D]["delaunay hypercube"]

        @test sum(geom.volumes[D].values) ≈ 1
    end

    @testset "Random Delaunay Hypercube" begin
        geom = geometries[D]["random delaunay hypercube"]

        @test sum(geom.volumes[D].values) ≈ 1
    end
end

@testset "Geometry ops D=$D P=$P R=$R" for D in 1:Dmax, P in (Pr, Dl), R in 0:D
    T = Float64

    @testset "$(geom.name)" for (geom_name, geom) in geometries[D]
        f0 = zeros(Fun{D,P,R,T}, geom.topo)
        f1 = ones(Fun{D,P,R,T}, geom.topo)
        fid = id(Fun{D,P,R,T}, geom.topo)
        funs = [f0, f1, fid]
        if R == 0 && P == Pr
            for d in 1:D
                if geom.topo.nvertices == 0
                    # Cannot deduce element type
                    xs = T[]
                else
                    xs = map(x -> x[d], geom.coords.values)
                end
                f = Fun{D,P,R}(geom.topo, xs)
                push!(funs, f)
            end
        end

        for f in funs
            h = hodge(Val(P), Val(R), geom)
            # Require Hodge operator to be at least positive semidefinite
            @test all(>=(0), h.values.diag)
            if geom_name ∉ ["orthogonal simplex", "standard hypercube",
                "delaunay hypercube"]
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
                for f in funs
                    cf = c * f
                    cf::Fun{D,P,R - 1,T}
                end
            end

            for f in funs
                l = laplace(Val(P), Val(R), geom)
                lf = l * f
                lf::Fun{D,P,R,T}
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

    @testset "$(geom.name)" for (geom_name, geom) in geometries[D]
        f0 = zeros(Fun{D,P,R,T}, geom.topo)
        f1 = ones(Fun{D,P,R,T}, geom.topo)
        fid = id(Fun{D,P,R,T}, geom.topo)
        fc = geom.coords
        funs = [f0, f1, fid, fc]

        for (fi, f) in enumerate(funs)

            # Interpolating at nodes must give nodal values
            if R == 0 && P == Pr
                for n in 1:(geom.topo.nvertices)
                    x = geom.coords.values[n]
                    if !(abs(evaluate(geom, f, x) - f.values[n]) + 1 ≈ 1)
                        @show D P R geom_name fi n
                    end
                    @test abs(evaluate(geom, f, x) - f.values[n]) + 1 ≈ 1
                end
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
                    elseif f == fid
                        # do nothing
                    elseif f == fc
                        @test evaluate(geom, f, x0) == x0
                    end
                end
            end
        end
    end
end

@testset "Discretize a function D=$D P=$P R=$R" for D in 1:Dmax
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
