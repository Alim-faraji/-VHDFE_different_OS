using VarianceComponentsHDFE
using Random, BenchmarkTools, Test
using JLD, SparseArrays


# Entry point for the PkgBenchmarks call.  Can split into different files later.
include("prepare_benchmark_data.jl")
# NOTE: Suite below can assume that the `benchmark/data/...` has been filled

settings_default = Settings()
settings_direct = Settings(lls_algorithm = DirectLLS())
settings_CMG = Settings(lls_algorithm = CMGPreconditionedLLS())

medium_data = load("data/medium_main.jld")
Xmedium_Laplacian = medium_data["X_Laplacian"]
Xmedium_GroundedLaplacian = medium_data["X_GroundedLaplacian"]

const SUITE = BenchmarkGroup()

idx = [1, 10000, 20000, 30000, 40000, 50000]

# Setup the benchmark suites for testing the default LSS algorithm
for i in idx
    SUITE["Default LSS", i] = @benchmarkable lss(settings_default.lls_algorithm, Xmedium_Laplacian, Xmedium_Laplacian[i,:], settings_default)
    # SUITE["CMG LSS", i] = @benchmarkable lss(settings_CMG.lls_algorithm, Xmedium_Laplacian, Xmedium_Laplacian[i,:], settings_CMG)
end
