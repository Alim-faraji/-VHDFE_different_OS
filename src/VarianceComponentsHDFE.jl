module VarianceComponentsHDFE

using DataDeps, CSV
using DataFrames, DataFramesMeta, Parameters
using LinearAlgebra, SparseArrays, Random, Statistics
using SparseArrays, IterativeSolvers, LightGraphs, VectorizedRoutines, CSVFiles, DataFramesMeta, Laplacians
using Distributions, Arpack

include("init.jl")
include("leave_out_correction.jl")
include("parameters_settings.jl")

export find_connected_set,prunning_connected_set,drop_single_obs, index_constr
export compute_movers, check_clustering, eff_res
export do_Pii, lincom_KSS, compute_matchid, leave_out_estimation
export Settings, JLAAlgorithm, ExactAlgorithm, AbstractLeverageAlgorithm

end
