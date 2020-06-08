using VarianceComponentsHDFE
using Random, BenchmarkTools, Test


# Entry point for the PkgBenchmarks call.  Can split into different files later.
include("prepare_benchmark_data.jl")



const SUITE = BenchmarkGroup()
# NOTE: Suite below can assume that the `benchmark/data/...` has been filled
SUITE["getlagged"] = @benchmarkable getlagged([1.0, 2.0, 3.0])
