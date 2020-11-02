# DDF.jl: Discrete Differential Forms

**A Julia package for discretizing functions via finite elements.**

The `DDF` package discretizes functions on arbitrary domains. The
domain is decomposed into simplices, and the function is represented
via a basis in each simplex. This is a form of a Finite Element
discretization. For example, a scalar function in two dimensions might
be represented via its values at the vertices. In between the vertices
it might be defined via linear interpolation.

The `DDF` package is based on
[FEEC](http://www-users.math.umn.edu/~arnold/), the Finite Element
Exterior Calculus. FEEC bears a certain similarity to
[DEC](https://en.wikipedia.org/wiki/Discrete_exterior_calculus), the
Discrete Exterior Calculus.

The design goals of the `DDF` package are:
- supports unstructured meshes (e.g. simplices)
- works in arbitrary dimensions
- supports higher order accurate discretizations
- is efficient enough for parallel/distributed calculations with large
  meshes with billions of elements
- is flexible and relatively easy to use so that it is useful for
  experimental mathematics

The `DDF` package is currently (2020-10-17) work in progress.
Functionality is missing, and the current API is too tedious (too many
details need to be specified explicitly).

## Table of Contents

```@contents
```

## Types

```@docs
Manifold
Op
Fun
```

```@docs
SparseOp
```

## Index

```@index
```
