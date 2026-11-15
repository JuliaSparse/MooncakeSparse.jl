# MooncakeSparse.jl

A sparse-matrix extension for the automatic differentiation library [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl).

## Supported Operations

- `mul!(C, A, B)`
- `mul!(C, A, B, α, β)`, for a `SparseMatrixCSC` `A`
- `dot(x, A, y)`
