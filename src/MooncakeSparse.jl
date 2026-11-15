module MooncakeSparse

export toarray, ldivwith, ldivwith!, rdivwith, rdivwith!

using LinearAlgebra
using LinearAlgebra: Hermitian, Symmetric, Adjoint, Transpose, AdjOrTrans, rmul!, ldiv!, rdiv!, dot, axpy!, axpby!, mul!
using LinearAlgebra: BlasFloat

using Base: IEEEFloat

using SparseArrays
using SparseArrays: SparseMatrixCSC, nzrange, rowvals, getcolptr, nonzeros

using Base.Order: Forward

using Mooncake
using Mooncake: @is_primitive, MinimalCtx, CoDual, Dual, NoRData, NoFData, primal, tangent,
                Tangent, FData, FriendlyTangentCache, AsCustomised,
                friendly_tangent_cache, tangent_to_friendly_internal!!, zero_rdata, arrayify

# The element types of Mooncake's rule for dense `mul!`.
@static if isdefined(Core, :BFloat16)
    const HalfFloat = Union{Float16, Core.BFloat16}
else
    const HalfFloat = Float16
end

# As in Mooncake's rule for dense `mul!`, a coefficient is a `Bool`, a float, or of the
# element type of the arrays.
const Coefficient{T} = Union{Bool, IEEEFloat, T}

const HermSparse{T, I} = Hermitian{T, SparseMatrixCSC{T, I}}
const SymSparse{T, I} = Symmetric{T, SparseMatrixCSC{T, I}}
const HermOrSymSparse{T, I} = Union{HermSparse{T, I}, SymSparse{T, I}}
const MaybeHermOrSymSparse{T, I} = Union{SparseMatrixCSC{T, I}, HermOrSymSparse{T, I}}

const ConjHermSparse{T, I} = Transpose{T, HermSparse{T, I}}
const ConjSymSparse{T, I} = Adjoint{T, SymSparse{T, I}}
const ConjHermOrSymSparse{T, I} = Union{ConjHermSparse{T, I}, ConjSymSparse{T, I}}
const HermOrSymOrConjSparse{T, I} = Union{HermOrSymSparse{T, I}, ConjHermOrSymSparse{T, I}}

const AdjSparse{T, I} = Adjoint{T, SparseMatrixCSC{T, I}}
const TransSparse{T, I} = Transpose{T, SparseMatrixCSC{T, I}}
const AdjOrTransSparse{T, I} = Union{AdjSparse{T, I}, TransSparse{T, I}}
const MaybeAdjOrTransSparse{T, I} = Union{SparseMatrixCSC{T, I}, AdjOrTransSparse{T, I}}

const ConjSparse{T, I} = Union{Adjoint{T, TransSparse{T, I}}, Transpose{T, AdjSparse{T, I}}}
const MaybeAdjOrTransOrConjSparse{T, I} = Union{MaybeAdjOrTransSparse{T, I}, ConjSparse{T, I}}

const DenseMatrix{T} = Union{StridedMatrix{T}, AdjOrTrans{T, <:StridedVecOrMat{T}}}
const DenseVecOrMat{T} = Union{DenseMatrix{T}, StridedVector{T}}

include("utils.jl")
include("friendly.jl")
include("matrix.jl")
include("register.jl")

end # module MooncakeSparse
