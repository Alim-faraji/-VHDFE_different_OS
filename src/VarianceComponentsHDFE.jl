module VarianceComponentsHDFE

# using DataDeps
using CSV
using DataFrames, DataFramesMeta, Parameters
using LinearAlgebra, SparseArrays, Random, Statistics
using SparseArrays, LightGraphs, VectorizedRoutines, CSVFiles, DataFramesMeta
using Distributions, Arpack
using LinearOperators, FastClosures, Krylov

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

function julia_main()::Cint
    try
        real_main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

function real_main()::Cint
    data = CSV.read(ARGS[1];header=false)
    id = data[:,1]
    firmid = data[:,2]
    y = data[:,4]
    controls = nothing
    println("successfully opened the CSV")
    settings = Settings(leverage_algorithm = JLAAlgorithm(num_simulations=parse(Int,ARGS[2])), person_effects=true, cov_effects=true)
    θFE, θPE, θCOV = compute_whole(y,id,firmid,controls,settings;verbose=false)
    println((θFE, θPE, θCOV))
    return 0
end

end
