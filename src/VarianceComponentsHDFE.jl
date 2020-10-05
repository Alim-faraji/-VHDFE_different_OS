module VarianceComponentsHDFE

# using DataDeps
using CSV
using DataFrames, DataFramesMeta, Parameters
using LinearAlgebra, SparseArrays, Random, Statistics
using SparseArrays, LightGraphs, VectorizedRoutines, CSVFiles, DataFramesMeta
using Distributions, Arpack
using LinearOperators, FastClosures, Krylov
using ArgParse

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

function parse_commandline()
    argparsesettings_obj = ArgParseSettings()

    @add_arg_table! argparsesettings_obj begin
        "path"
            help = "path to CSV file containing data"
            required = true
        "--id"
            help = "column index in CSV file for id"
            arg_type = Int
            default = 1
        "--firmid"
            help = "column index in CSV file for firmid"
            arg_type = Int
            default = 2
        "--y"
            help = "column index in CSV file for y"
            arg_type = Int
            default = 4
        "--algorithm"
            help = "type of algorithm: exact or JLA"
            arg_type = String
            default = "Exact"
        "--simulations"
            help = "number of simulations in the JLA algorithm"
            arg_type = Int
            default = 100
        "--header"
            help = "CSV file contains header"
            action = :store_true
        #=
        "--person_effects"
            help = "compute person effects"
            action = :store_true
        "--cov_effects"
            help = "compute cov effects"
            action = :store_true
        =#
        "--write_CSV"
            help = "write output to a CSV"
            action = :store_true
        "--output_path"
            help = "path to output CSV"
            arg_type = String
            default = "VarianceComponents.csv"
    end

    return parse_args(argparsesettings_obj)
end

function julia_main()::Cint
    try
        real_main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

function real_main()

    println("Number of threads: $(Threads.nthreads())")

    parsed_args = parse_commandline()

    path = parsed_args["path"]
    header = parsed_args["header"]
    id_idx = parsed_args["id"]
    firmid_idx = parsed_args["firmid"]
    y_idx = parsed_args["y"]
    algorithm = parsed_args["algorithm"]
    # person_effects = parsed_args["person_effects"]
    # cov_effects = parsed_args["cov_effects"]
    simulations = parsed_args["simulations"]

    data  = DataFrame!(CSV.File(path; header=header))
    id = data[:,id_idx]
    firmid = data[:,firmid_idx]
    y = data[:,y_idx]

    controls = nothing

    if algorithm == "Exact"
        settings = Settings(leverage_algorithm = ExactAlgorithm(), person_effects=true, cov_effects=true)
    else
        settings = Settings(leverage_algorithm = JLAAlgorithm(num_simulations=simulations), person_effects=true, cov_effects=true)
    end

    θFE, θPE, θCOV, obs, β, Dalpha, Fpsi, Pii, Bii_pe, Bii_fe, Bii_cov = compute_whole(y,id,firmid,controls,settings;verbose=true)

    println("Bias-Corrected Variance Components:")
    println("Bias-Corrected Variance of Firm Effects: $θFE")
    println("Bias-Corrected Variance of Person Effects: $θPE")
    println("Bias-Corrected Covariance of Firm-Person Effects: $θCOV")

    if parsed_args["write_CSV"]

        y_output = y[obs]
        id_output = id[obs]
        firmid_output = firmid[obs]

        max_length = length(obs)

        output = DataFrame(observation = obs,
                           worker_id = id_output,
                           firm_id = firmid_output,
                           y = y_output,
                           beta = vcat(β,missings(max(max_length-length(β),0))),
                           D_alpha = Dalpha,
                           F_psi = Fpsi,
                           Pii = Pii,
                           Bii_pe = Bii_pe,
                           Bii_fe = Bii_fe,
                           Bii_cov = Bii_cov,
                           variance_comp_firm_effects = [θFE; missings(max_length-1)],
                           variance_comp_person_effects = [θPE; missings(max_length-1)],
                           covariance_comp_effects = [θCOV; missings(max_length-1)])
        output_path = parsed_args["output_path"]
        CSV.write(output_path,output)
    end
    return
end

end
