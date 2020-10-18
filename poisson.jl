using Revise

using DDF
using LinearAlgebra

# Base.chop(x::Number) = abs2(x) < eps(x)^2 ? zero(x) : x

D = 2
S = Float64
T = Float64

mfd0 = simplex_manifold(Val(D), S);
mfd1 = refined_manifold(mfd0);
mfd2 = refined_manifold(mfd1);
mfd = mfd2;

################################################################################

# L u == ρ
# B d u == 0
# 
# d u - f == 0
# δ f == ρ
# B f == 0

ρ = unit(Fun{D,Pr,0,D,S,T}, mfd, nsimplices(mfd, 0));
u₀ = zero(Fun{D,Pr,0,D,S,T}, mfd);

d = deriv(Val(Pr), Val(0), mfd);
δ = coderiv(Val(Pr), Val(1), mfd);

# A mask for all simplices
χ = Fun{D,Pr,D,D,S,Bool}(mfd, ones(Bool, nsimplices(mfd, D)));
# A mask for all boundary faces
∂χ = map(isodd, boundary(χ));
# A mask for all boundary vertices
∂0 = Fun{D,Pr,0,D,S,Bool}(mfd,
                          map(≠(0),
                              map(Bool, mfd.lookup[(0, D - 1)]).op * ∂χ.values))
# Boundary operator for vertices
B0 = Op{D,Pr,0,Pr,0}(mfd, Diagonal(∂0.values));

N0 = zero(Op{D,Pr,0,Pr,0,Bool}, mfd);
N1 = zero(Op{D,Pr,1,Pr,1,Bool}, mfd);
E0 = one(Op{D,Pr,0,Pr,0,Bool}, mfd);
E1 = one(Op{D,Pr,1,Pr,1,Bool}, mfd);
n0 = zero(Fun{D,Pr,0,D,S,Bool}, mfd);
n1 = zero(Fun{D,Pr,1,D,S,Bool}, mfd);

N01 = zero(Op{D,Pr,0,Pr,1,Bool}, mfd);
N10 = zero(Op{D,Pr,1,Pr,0,Bool}, mfd);

E = [E0.values N01.values; N10.values E1.values];
@assert E * E == E

# We want to solve this:
#     [0  δ] [u] = [ρ]
#     [d -1] [f] = [0]
#     A x == b
# but need to handle boundary conditions. The boundary conditions are:
#     [B  0] [u] = [u₀]
#     [0  0] [f] = [0 ]
#     B x == B c
# Projecting these out of the linear system:
#     (1-B) A x == (1-B) b
#     A′ x = b′

A = [N0.values δ.values; d.values -E1.values];
b = [ρ.values; n1.values];

B = [B0.values N01.values; N10.values N1.values];
@assert B * B == B
c = [u₀.values; n1.values];

A′ = (E - B) * A + B
b′ = (E - B) * b + B * c

x = A′ \ b′

u = Fun{D,Pr,0,D,S,T}(mfd, @view x[1:nsimplices(mfd, 0)])
f = Fun{D,Pr,1,D,S,T}(mfd, @view x[(nsimplices(mfd, 0) + 1):end])

norm((E0-B0) * (laplace(u) - ρ), Inf)
norm(B0 * u - u₀, Inf)

err = (E0-B0) * (laplace(u) - ρ) + B0 * u - u₀;

################################################################################

using AbstractPlotting
using GLMakie
using StaticArrays

coordinates = [mfd.coords[0][i][d] for i in 1:nsimplices(mfd, 0), d in 1:D]
connectivity = [SVector{D + 1}(i
                               for i in sparse_column_rows(mfd.simplices[D], j))
                for j in 1:size(mfd.simplices[D], 2)]
connectivity = [connectivity[i][n]
                for i in 1:nsimplices(mfd, D), n in 1:(D + 1)]
# color = ∂0.values
# color = ρ.values
# color = u.values
color = err.values

scene = Scene()
poly!(scene, coordinates, connectivity, color=color, strokecolor=(:black, 0.6),
      strokewidth=4)
scale!(scene, 1, 1)
