# Operators


## Types of operators
Operator is a quite general concept and is everything that makes modifications
to the system matrix, hence classical represenations of weak discretisations of differential operators,
penalisations for boundary conditions or global constraints, or stabilisation terms.

The three most important operator classes are:
- NonlinearOperator (e.g. the convection term in a Navier-Stokes problem)
- BilinearOperator (e.g. the Laplacian in a Poisson problem)
- LinearOperator (e.g. the right-hand side in a Poisson or Navier-Stokes problem)

To assing boundary conditions or global constraints there are three possibilities:
- InterpolateBoundaryData
- HomogeneousData
- FixDofs




## Entities and Regions

Each operator assembles on certain entities of the mesh, the default is a cell-wise
assembly. Most operators have the entities kwarg to changes that. Restrictions to subsets
of the entities can be made via the regions kwarg.

| Entities         | Description                                                      |
| :--------------- | :--------------------------------------------------------------- |
| AT_NODES         | interpolate at vertices of the mesh (only for H1-conforming FEM) |
| ON_CELLS         | assemble/interpolate on the cells of the mesh                  |
| ON_FACES         | assemble/interpolate on all faces of the mesh                  |
| ON_IFACES        | assemble/interpolate on the interior faces of the mesh         |
| ON_BFACES        | assemble/interpolate on the boundary faces of the mesh         |
| ON_EDGES (*)     | assemble/interpolate on all edges of the mesh (in 3D)          |
| ON_BEDGES (*)    | assemble/interpolate on the boundary edges of the mesh (in 3D) |

!!! note
    (*) = only reasonable in 3D and still experimental, might have some issues
