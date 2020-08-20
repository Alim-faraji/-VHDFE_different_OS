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
        "--person_effects"
            help = "compute person effects"
            action = :store_true
        "--cov_effects"
            help = "compute cov effects"
            action = :store_true
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
    parsed_args = parse_commandline()

    path = parsed_args["path"]
    header = parsed_args["header"]
    id_idx = parsed_args["id"]
    firmid_idx = parsed_args["firmid"]
    y_idx = parsed_args["y"]
    algorithm = parsed_args["algorithm"]
    person_effects = parsed_args["person_effects"]
    cov_effects = parsed_args["cov_effects"]
    simulations = parsed_args["simulations"]

    data = CSV.read(path;header=header)
    id = data[:,id_idx]
    firmid = data[:,firmid_idx]
    y = data[:,y_idx]

    controls = nothing

    if algorithm == "Exact"
        settings = Settings(leverage_algorithm = ExactAlgorithm(), person_effects=person_effects, cov_effects=cov_effects)
    else
        settings = Settings(leverage_algorithm = JLAAlgorithm(num_simulations=simulations), person_effects=person_effects, cov_effects=cov_effects)
    end

    θFE, θPE, θCOV = compute_whole(y,id,firmid,controls,settings;verbose=false)
    println((θFE, θPE, θCOV))

    if parsed_args["write_CSV"]
        output = DataFrame(firm_effects = [ΘFE], person_effects = [ΘPE], cov_effects = [ΘCOV])
        output_path = parsed_args["output_path"],
        CSV.write(output_path,output)
    end
    return
end

end
