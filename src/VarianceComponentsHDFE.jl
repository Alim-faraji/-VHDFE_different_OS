module VarianceComponentsHDFE

using DataDeps, CSV
using DataFrames, DataFramesMeta, Parameters
using LinearAlgebra, SparseArrays, Random, Statistics
using SparseArrays, LightGraphs, VectorizedRoutines, CSVFiles, DataFramesMeta
using Distributions, Arpack
using LinearOperators, FastClosures, Krylov
using IterativeSolvers, AlgebraicMultigrid

include("init.jl")
include("leave_out_correction.jl")
include("parameters_settings.jl")
include("laplacians/Laplacians.jl")

using .Laplacians
include("solvers.jl")

export find_connected_set,prunning_connected_set,drop_single_obs, index_constr
export compute_movers, check_clustering, eff_res
export do_Pii, lincom_KSS, compute_matchid, leave_out_estimation, compute_whole
export Settings, JLAAlgorithm, ExactAlgorithm, AbstractLeverageAlgorithm

# Exporting these functions for ease of benchmarking/testing
export computeLDLinv, approxcholOperator, approxcholSolver

end
