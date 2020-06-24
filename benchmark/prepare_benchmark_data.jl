using SparseArrays, LinearAlgebra
using JLD, CSV, DataFrames, DataFramesMeta, DataDeps

# add in code to generate .jld, .csv, etc. in the benchmark/data directory
# This should only generate if the file doesn't exist, or if this `force_generate = true`
force_generate = false

pkg_dir = pkgdir(VarianceComponentsHDFE)

# Obtain environment variables
run_large_benchmark = get(ENV, "VCHDFE_LARGE_BENCHMARK", "false")  == "true" ? true : false

## Medium-Sized Network Generator
function rademacher!(R; demean = false)
    R .= R - (R .== 0)

    if demean == true
        R .= R .- mean(R, dims = 2)
    end
    return nothing
end

function compute_X_No_Controls(data)
    id = data.id
    firmid = data.firmid
    y = data.y

    NT = size(y,1);
    J = maximum(firmid);
    N = maximum(id);
    K = 0
    nparameters = N + J + K

    #Worker Dummies
    D = sparse(collect(1:NT),id,1);

    #Firm Dummies
    F = sparse(collect(1:NT),firmid,1);

    # N+J x N+J-1 restriction matrix
    S= sparse(1.0I, J-1, J-1);
    S=vcat(S,sparse(-zeros(1,J-1)));
    X_Laplacian = hcat(D, -F)
    X_GroundedLaplacian = hcat(D, -F*S)
    S_xx = Symmetric(X_GroundedLaplacian'*X_GroundedLaplacian)

    return X_Laplacian, X_GroundedLaplacian, S_xx
end

function compute_X_Controls(data)
    id = data.id
    firmid = data.firmid
    y = data.y


    NT=size(y,1);
    J=maximum(firmid);
    N=maximum(id);
    K = 2
    nparameters = N + J + K

    #Worker Dummies
    D=sparse(collect(1:NT),id,1);

    #Firm Dummies
    F=sparse(collect(1:NT),firmid,1);

    # N+J x N+J-1 restriction matrix
    S= sparse(1.0I, J-1, J-1);
    S=vcat(S,sparse(-zeros(1,J-1)));

    #Assuming columns 5 and 6 are the controls
    controls = hcat(data.control1[id], data.control2[id])

    Xcontrols = hcat(D,F*S,controls)
    S_xx = Symmetric(Xcontrols'*Xcontrols)

    return Xcontrols, S_xx
end

if ~isfile(pkg_dir*"/benchmark/data/medium_main.jld") || force_generate
    data = CSV.read(datadep"VarianceComponentsHDFE/medium_nocontrols_pruned.csv"; header=true)
    X_Laplacian, X_GroundedLaplacian, S_xx = compute_X_No_Controls(data)
    save(pkg_dir*"/benchmark/data/medium_main.jld", "X_Laplacian", X_Laplacian, "X_GroundedLaplacian", X_GroundedLaplacian, "S_xx", S_xx)
end

if ~isfile(pkg_dir*"/benchmark/data/medium_controls_main.jld") || force_generate
    data = CSV.read(datadep"VarianceComponentsHDFE/medium_controls_pruned.csv"; header=true)
    Xcontrols, S_xx = compute_X_Controls(data)
    save(pkg_dir*"/benchmark/data/medium_controls_main.jld", "Xcontrols", Xcontrols, "S_xx", S_xx)
end

if run_large_benchmark && (~isfile(pkg_dir*"/benchmark/data/large_main.jld") || force_generate)
    data = CSV.read(datadep"VarianceComponentsHDFE/large_nocontrols_pruned.csv"; header=true)
    X_Laplacian, X_GroundedLaplacian, S_xx = compute_X_No_Controls(data)
    save(pkg_dir*"/benchmark/data/large_main.jld", "X_Laplacian", X_Laplacian, "X_GroundedLaplacian", X_GroundedLaplacian, "S_xx", S_xx)
 end

 if run_large_benchmark && (~isfile(pkg_dir*"/benchmark/data/large_controls_main.jld") || force_generate)
     data = CSV.read(datadep"VarianceComponentsHDFE/large_controls_pruned.csv"; header=true)
     Xcontrols, S_xx = compute_X_Controls(data)
     save(pkg_dir*"/benchmark/data/large_controls_main.jld", "Xcontrols", Xcontrols, "S_xx", S_xx)
 end
