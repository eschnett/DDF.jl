using DDF

using SparseArrays
using StaticArrays
using Test



@testset "DManifold D=$D" for D in 0:Dmax

    function checkboundary2(mf::DManifold{D}) where {D}
        for R in 2:D
            boundary2 = dropzeros(mf.boundaries[R-1] * mf.boundaries[R])
            @test nnz(boundary2) == 0
        end
    end

    mf0 = DManifold(Val(D))
    for R in 0:D
        @test size(Val(R), mf0) == 0
    end
    checkboundary2(mf0)

    mf1 = DManifold(DSimplex(SVector{D+1}(1:D+1)))
    for R in 0:D
        @test size(R, mf1) == binomial(D+1, R+1)
    end
    checkboundary2(mf1)

    # if D > 0x
    if D > 1
        # boundary of the simplex
        if D == 1
            # TODO: Provide a constructor for manifolds with
            # disconnected vertices
            mf2 = DManifold([DSimplex{D,Int}(SVector(i))
                             for i in 1:mf1.nvertices])
        else
            mf2 = DManifold(mf1.simplices[D-1])
        end
        @test ndims(mf2) == D-1
        checkboundary2(mf2)
    end

    if D == 2
        # a Möbius strip (arXiv:1103.3076v2 [cs.NA], figure 7)
        mf2 = DManifold(SVector{D+1}.([(1, 2, 4),
                                       (1, 4, 6),
                                       (4, 3, 6),
                                       (6, 3, 5),
                                       (3, 1, 5),
                                       (1, 2, 5)]))
        @test ndims(mf2) == D
        @test size(Val(0), mf2) == 6
        @test size(Val(1), mf2) == 12
        @test size(Val(2), mf2) == 6

        checkboundary2(mf2)
    end

    if D == 2
        # a topological sphere (surface of a tetrahedron)
        mf2 = DManifold(SVector{D+1}.([(1, 2, 3),
                                       (1, 3, 4),
                                       (1, 4, 2),
                                       (2, 3, 4)]))
        @test ndims(mf2) == D
        @test size(Val(0), mf2) == 4
        @test size(Val(1), mf2) == 6
        @test size(Val(2), mf2) == 4

        checkboundary2(mf2)
    end

    if D == 1
        # a line
        p0, p2, p1 = 1:3
        mf3 = DManifold(SVector{D+1}.([(p0, p1), (p2, p1)]))
        @test ndims(mf3) == D
        checkboundary2(mf3)
    end

    if D == 2
        # a square
        p00, p02, p20, p22,
        p11 = 1:5
        mf4 = DManifold(SVector{D+1}.([(p00, p02, p11),
                                       (p02, p22, p11),
                                       (p22, p20, p11),
                                       (p20, p00, p11)]))
        @test ndims(mf4) == D
        checkboundary2(mf4)
    end

    if D == 3
        # a cube
        p000, p002, p020, p022, p200, p202, p220, p222, 
        p110, p112, p101, p121, p011, p211, 
        p111 = 1:15
        mf5 = DManifold(SVector{D+1}.([(p000, p020, p110, p111),
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
                                       (p220, p200, p211, p111)]))
        @test ndims(mf5) == D
        checkboundary2(mf5)
    end

    # a hypercube
    mf6 = hypercube_manifold(Val(D))
    @test ndims(mf6) == D
    checkboundary2(mf6)

    # TODO: Provide a constructor for manifolds with
    # disconnected vertices
    # if D > 0
    if D > 1
        # "boundary" of the hypercube (this is not a manifold)
        mf7 = DManifold(mf6.simplices[D-1])
        @test ndims(mf7) == D-1
        checkboundary2(mf7)
    end
end
