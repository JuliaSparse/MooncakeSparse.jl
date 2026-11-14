using MooncakeSparse
using MooncakeSparse: toarray
using Mooncake: Config, prepare_derivative_cache, prepare_gradient_cache,
                zero_tangent, value_and_derivative!!, value_and_gradient!!,
                Tangent
using Mooncake
using Mooncake.TestUtils: test_rule
using LinearAlgebra
using SparseArrays
using SparseArrays: nonzeros
using Random: randn!, Xoshiro
using Test

function randtangent(A)
    dA = copy(A)
    return randtangent!(dA)
end

function randtangent(A::Adjoint)
    return adjoint(randtangent(parent(A)))
end

function randtangent(A::Transpose)
    return transpose(randtangent(parent(A)))
end

function randtangent!(A::AbstractArray)
    return randn!(A)
end

function randtangent!(A::SparseMatrixCSC)
    randn!(nonzeros(A))
    return A
end

function randtangent!(A::Union{Hermitian,Symmetric})
    randtangent!(parent(A))
    return A
end

function maybefriendly(x, g)
    if g isa Tangent
        return toarray(x, g)
    end

    return g
end

function flat(A)
    return A
end

function flat(A::Union{Symmetric, Hermitian, Adjoint, Transpose})
    return flat(parent(A))
end

function dotflat(A, B)
    return dot(flat(A), flat(B))
end

function testadjoint(f, args...; rtol=1e-4)
    config = Config(friendly_tangents=true)

    fwd_cache = prepare_derivative_cache(f, args...; config)
    df = zero_tangent(f)

    tangents = map(randtangent, args)
    _, dy = value_and_derivative!!(fwd_cache, (f, df), zip(args, tangents)...)

    rev_cache = prepare_gradient_cache(f, args...; config)
    _, (_, gradients...) = value_and_gradient!!(rev_cache, f, args...)

    arrays = map(maybefriendly, args, gradients)
    return isapprox(dy, sum(real ∘ splat(dotflat), zip(arrays, tangents)); rtol)
end

n = 100
k = 10
p = 0.3

@testset "Adjoint Consistency" begin
    for T in (Float64, ComplexF64)
        @testset "$T" begin
            AU = sprandn(T, n, n, p) + n * I
            AH = Hermitian(AU, :L)
            AS = Symmetric(AU, :L)

            x = randn(T, n)
            y = randn(T, n)

            X = randn(T, k, n)
            Y = randn(T, n, k)

            @testset "lmul" begin
                for L in (AU, AU', transpose(AU), transpose(AU)', AH, AS, transpose(AH), AS'), R in (X', Y, y)
                    @test testadjoint((L, R) -> real(sum(L * R)), L, R)
                end
            end

            @testset "rmul" begin
                for L in (X, Y', y'), R in (AU, AU', transpose(AU), transpose(AU)')
                    @test testadjoint((L, R) -> real(sum(L * R)), L, R)
                end
            end

            @testset "dot (3-arg)" begin
                for A in (AU, AU', transpose(AU), transpose(AU)', AH, AS, transpose(AH), AS')
                    @test testadjoint((x, A, y) -> real(dot(x, A, y)), x, A, y)
                end
            end

            @testset "ldivwith" begin
                for A in (AU, AH, AS), R in (Y, y)
                    @test testadjoint((A, R) -> real(sum(ldivwith(A, lu(Matrix(A)), R))), A, R)
                end
            end

            # rdivwith tests disabled due to Julia stdlib bug:
            # mul!(C, Y, Hermitian{SparseMatrixCSC}) ignores the Hermitian wrapper
            # https://github.com/JuliaSparse/SparseArrays.jl/issues/688
        end
    end
end

@testset "mul! (5-arg)" begin
    rng = Xoshiro(123)

    @testset "$T $(ndims_B == 1 ? "vector" : "matrix") α=$(typeof(α)) β=$(typeof(β))" for T in (Float64, Float32, ComplexF64, ComplexF32),
            ndims_B in (1, 2),
            (α, β) in (
                (true, false),
                (true, true),
                (randn(rng, T), randn(rng, T)),
                (real(T)(0.7), real(T)(-0.4)),
                (T(1.3), false),
                # coefficients of another precision than the arrays
                (real(T) == Float64 ? Float32 : Float64).((0.7, -0.4)),
            )

        A = sprand(rng, T, 7, 5, 0.4)
        B = ndims_B == 1 ? randn(rng, T, 5) : randn(rng, T, 5, 3)
        C = ndims_B == 1 ? randn(rng, T, 7) : randn(rng, T, 7, 3)
        test_rule(rng, mul!, C, A, B, α, β; is_primitive=true, perf_flag=:none)
    end

    # Finite differences are not reliable in half precision, so compare with Float64 instead.
    @testset "Float16 agrees with Float64 (α=$α β=$β)" for (α, β) in (
            (true, false), (true, true), (0.7, 0.3), (1.3, false)
        )

        A = sprand(rng, 7, 5, 0.4)
        B = randn(rng, 5, 3)
        C = randn(rng, 7, 3)
        function loss(A, B, C, α, β)
            C = copy(C)
            mul!(C, A, B, α, β)
            return sum(abs2, C)
        end
        to16(x) = x isa Bool ? x : Float16(x)
        args64 = (A, B, C, α, β)
        args16 = (Float16.(A), Float16.(B), Float16.(C), to16(α), to16(β))
        cache64 = prepare_gradient_cache(loss, args64...)
        cache16 = prepare_gradient_cache(loss, args16...)
        _, (_, dA64, dB64, dC64) = value_and_gradient!!(cache64, loss, args64...)
        _, (_, dA16, dB16, dC16) = value_and_gradient!!(cache16, loss, args16...)
        @test dA16.fields.nzval ≈ dA64.fields.nzval rtol = 1e-2
        @test dB16 ≈ dB64 rtol = 1e-2
        @test dC16 ≈ dC64 rtol = 1e-2 atol = 1e-2
    end
end

