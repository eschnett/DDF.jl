# Literature

- G. Westendorp, "A formula for the N-circumsphere of an N-simplex",
  <https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm>,
  April 2013.

- Gerard Westendorp, "Space-time triangles",
  <https://westy31.home.xs4all.nl/SpaceTimeTriangles/Space_Time_Triangles.html>.

- Charles G. Gunn, "Course notes: Geometric Algebra for Computer
  Graphics", SIGGRAPH 2019,
  <https://bivector.net/PROJECTIVE_GEOMETRIC_ALGEBRA.pdf>.

- <https://cseweb.ucsd.edu/classes/fa17/cse252A-a/lec4.pdf>.

- Anil N. Hirani, Kaushik Kalyanaraman, Evan B. VanderZee, "Delaunay
  Hodge Star", [arXiv:1204.0747v4
  [cs.CG]](https://arxiv.org/abs/1204.0747): Delaunay hodge,
  conditions for well-centred meshes.

- Volker Springel, "E pur si muove: Galiliean-invariant cosmological
  hydrodynamical simulations on a moving mesh", [arXiv:0901.4107
  [astro-ph.CO]](https://arxiv.org/abs/0901.4107).

- Michael Reed, "Differential geometric algebra with Leibniz and
  Grassmann", <https://crucialflow.com/grassmann-juliacon-2019.pdf>.



- WriteVTK.jl



[Glitter](<https://en.wikipedia.org/wiki/Glitter>)

... and [Glitterati](https://songmeanings.com/songs/view/2890/).



# DA

```
L u == ρ
B d u == 0

d u - f == 0
δ f == ρ
B f == 0
```


Can we rewrite this without δ or ⋆?

- DA: need to remove harmonic forms from f: P(1-H) f. only for strong
  form?

- DA: R[u]=0: von Neumann bc
      R[u]=D-1: Dirichlet bc

- DA: mixed weak formulations have no hodge dual
-     magnetic bc: R=1   (or R=2?)
-     electric bc: R=2   (or R=1?)

- DA: Hodge Laplacian is always well posed! choose complexes (and
      respective basis functions), then solve in the discrete with the
      same mixed weak formulation.

- DA: trace is projection onto boundary; trace maps D-dim form onto
      (D-1)-dim form

- DA: 0-forms: naturally piecewise continuous (FE), polynomial
      D-forms: naturally piecewise discontinuous (DG), polynomial
      - DOFs need to be located on either of vertices, edges, faces,
        etc.
      - must be unisolvent (be a basis?)
