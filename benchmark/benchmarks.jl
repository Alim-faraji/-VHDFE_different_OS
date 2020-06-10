using VarianceComponentsHDFE
using Random, BenchmarkTools, Test
using JLD, SparseArrays


# Entry point for the PkgBenchmarks call.  Can split into different files later.
include("prepare_benchmark_data.jl")

settings_default = Settings()
settings_direct = Settings(lls_algorithm = DirectLLS())
settings_CMG = Settings(lls_algorithm = CMGPreconditionedLLS())

medium_data = load("data/medium_main.jld")
Xmedium_Laplacian = medium_data["X_Laplacian"]
Xmedium_GroundedLaplacian = medium_data["X_GroundedLaplacian"]

const SUITE = BenchmarkGroup()
# NOTE: Suite below can assume that the `benchmark/data/...` has been filled
# SUITE["getlagged"] = @benchmarkable getlagged([1.0, 2.0, 3.0])

SUITE["Default LSS Algorithm"] = @benchmarkable lss(settings_default.lls_algorithm, Xmedium_Laplacian, Xmedium_Laplacian[1,:], settings_default)

# The direct method gives a singular value exception
# SUITE["Direct LSS Algorithm"] = @benchmarkable lss(settings_direct.lls_algorithm, Xmedium_Laplacian, Xmedium_Laplacian[1,:], settings_direct)
