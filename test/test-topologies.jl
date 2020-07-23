using DDF

using SparseArrays
using StaticArrays
using Test



@testset "Topology D=$D" for D = 0:Dmax
    function checkboundary2(topo::Topology{D}) where {D}
        for R = 2:D
            boundary2 = dropzeros(topo.boundaries[R-1] * topo.boundaries[R])
            @test nnz(boundary2) == 0
        end
    end

    topo0 = Topology(Val(D))
    for R = 0:D
        @test size(Val(R), topo0) == 0
    end
    checkboundary2(topo0)

    topo1 = Topology(Simplex(SVector{D + 1}(1:D+1)))
    for R = 0:D
        @test size(R, topo1) == binomial(D + 1, R + 1)
    end
    checkboundary2(topo1)

    if D > 1
        # boundary of the simplex
        topo2 = Topology("simpliex boundary", topo1.simplices[D-1])
        @test ndims(topo2) == D - 1
        checkboundary2(topo2)
    end

    if D == 2
        # a Möbius strip (arXiv:1103.3076v2 [cs.NA], figure 7)
        topo2 = Topology(
            "Möbius strip",
            SVector{
                D + 1,
            }.([(1, 2, 4), (1, 4, 6), (4, 3, 6), (6, 3, 5), (3, 1, 5), (1, 2, 5)]),
        )
        @test ndims(topo2) == D
        @test size(Val(0), topo2) == 6
        @test size(Val(1), topo2) == 12
        @test size(Val(2), topo2) == 6

        checkboundary2(topo2)
    end

    if D == 2
        # a topological sphere (surface of a tetrahedron)
        topo2 = Topology(
            "tetrahedron surface",
            SVector{D + 1}.([(1, 2, 3), (1, 3, 4), (1, 4, 2), (2, 3, 4)]),
        )
        @test ndims(topo2) == D
        @test size(Val(0), topo2) == 4
        @test size(Val(1), topo2) == 6
        @test size(Val(2), topo2) == 4

        checkboundary2(topo2)
    end

    if D == 1
        # a line
        p0, p2, p1 = 1:3
        topo3 = Topology("line", SVector{D + 1}.([(p0, p1), (p2, p1)]))
        @test ndims(topo3) == D
        checkboundary2(topo3)
    end

    if D == 2
        # a square
        p00, p02, p20, p22, p11 = 1:5
        topo4 = Topology(
            "square",
            SVector{
                D + 1,
            }.([(p00, p02, p11), (p02, p22, p11), (p22, p20, p11), (p20, p00, p11)]),
        )
        @test ndims(topo4) == D
        checkboundary2(topo4)
    end

    if D == 3
        # a cube
        p000,
        p002,
        p020,
        p022,
        p200,
        p202,
        p220,
        p222,
        p110,
        p112,
        p101,
        p121,
        p011,
        p211,
        p111 = 1:15
        topo5 = Topology(
            "cube",
            SVector{
                D + 1,
            }.([
                (p000, p020, p110, p111),
                (p020, p220, p110, p111),
                (p220, p200, p110, p111),
                (p200, p000, p110, p111),
                (p002, p022, p112, p111),
                (p022, p222, p112, p111),
                (p222, p202, p112, p111),
                (p202, p002, p112, p111),
                (p000, p002, p101, p111),
                (p002, p202, p101, p111),
                (p202, p200, p101, p111),
                (p200, p000, p101, p111),
                (p020, p022, p121, p111),
                (p022, p222, p121, p111),
                (p222, p220, p121, p111),
                (p220, p020, p121, p111),
                (p000, p002, p011, p111),
                (p002, p022, p011, p111),
                (p022, p020, p011, p111),
                (p020, p000, p011, p111),
                (p200, p202, p211, p111),
                (p202, p222, p211, p111),
                (p222, p220, p211, p111),
                (p220, p200, p211, p111),
            ]),
        )
        @test ndims(topo5) == D
        checkboundary2(topo5)
    end

    # a hypercube
    topo6 = hypercube_manifold(Val(D))
    @test ndims(topo6) == D
    checkboundary2(topo6)

    # TODO: Provide a constructor for manifolds with
    # disconnected vertices
    # if D > 0
    if D > 1
        # "boundary" of the hypercube (this is not a manifold)
        topo7 = Topology("D=$D hypercube boundary", topo6.simplices[D-1])
        @test ndims(topo7) == D - 1
        checkboundary2(topo7)
    end
end
