using SparseArrays, LinearAlgebra
using JLD, CSV, DataFrames, DataFramesMeta, DataDeps

# add in code to generate .jld, .csv, etc. in the benchmark/data directory
# This should only generate if the file doesn't exist, or if this `force_generate = true`
force_generate = false

## Medium-Sized Network Generator
function rademacher!(R; demean = false)
    R .= R - (R .== 0)

    if demean == true
        R .= R .- mean(R, dims = 2)
    end
    return nothing
end

function compute_X_No_Controls(data)
    data = DataFrame(id = data[:,1], firmid = data[:,3], year = data[:,2], y = data[:,4] )
    sort!(data, [:id, :year])

    #step 1) LCS
    y = data.y
    id = data.id
    firmid = data.firmid
    data_akm = find_connected_set(y,id,firmid; verbose=true)

    #Step 2) Prunning Articulations
    y = data_akm.y
    firmid = data_akm.firmid
    id = convert(Array{Int64,1},data_akm.id)
    kss_data =  prunning_connected_set(y,id,firmid, data_akm.obs_id; verbose=true)

    #Step 3) Pruning Single Obs Workers
    kss_data = drop_single_obs(kss_data.y, kss_data.id, kss_data.firmid, kss_data.obs_id)

    y = kss_data.y
    firmid = kss_data.firmid

    temp = [data.firmid[x] for x in kss_data.obs_id]
    firmid = indexin(temp,unique(sort(temp)))
    id = convert(Array{Int64,1},kss_data.id)

    NT=size(y,1);
    J=maximum(firmid);
    N=maximum(id);
    K = 0
    nparameters = N + J + K

    #Worker Dummies
    D=sparse(collect(1:NT),id,1);

    #Firm Dummies
    F=sparse(collect(1:NT),firmid,1);

    # N+J x N+J-1 restriction matrix
    S= sparse(1.0I, J-1, J-1);
    S=vcat(S,sparse(-zeros(1,J-1)));

    X_Laplacian = hcat(D, -F)
    X_GroundedLaplacian = hcat(D, -F*S)

    S_xx = Symmetric(X_GroundedLaplacian'*X_GroundedLaplacian)

    return X_Laplacian, X_GroundedLaplacian, S_xx
end

function compute_X_Controls(data)
    data = DataFrame(id = data[:,1], firmid = data[:,3], year = data[:,2], y = data[:,4], control1 =data[:,5], control2=data[:,6] )
    sort!(data, [:id, :year])

    #step 1) LCS
    y = data.y
    id = data.id
    firmid = data.firmid
    data_akm = find_connected_set(y,id,firmid; verbose=true)

    #Step 2) Prunning Articulations
    y = data_akm.y
    firmid = data_akm.firmid
    id = convert(Array{Int64,1},data_akm.id)
    kss_data =  prunning_connected_set(y,id,firmid, data_akm.obs_id; verbose=true)

    #Step 3) Pruning Single Obs Workers
    kss_data = drop_single_obs(kss_data.y, kss_data.id, kss_data.firmid, kss_data.obs_id)

    y = kss_data.y
    firmid = kss_data.firmid

    temp = [data.firmid[x] for x in kss_data.obs_id]
    firmid = indexin(temp,unique(sort(temp)))
    id = convert(Array{Int64,1},kss_data.id)

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
    controls = hcat(data.control1[kss_data.obs_id], data.control2[kss_data.obs_id])
    
    Xcontrols = hcat(D,F*S,controls)
    S_xx = Symmetric(Xcontrols'*Xcontrols)

    return Xcontrols, S_xx
end

if ~isfile("data/medium_main.jld") || force_generate
    data = CSV.read(datadep"VarianceComponentsHDFE/medium_main.csv"; header=false)
    X_Laplacian, X_GroundedLaplacian, S_xx = compute_X_No_Controls(data)
    save("data/medium_main.jld", "X_Laplacian", X_Laplacian, "X_GroundedLaplacian", X_GroundedLaplacian, "S_xx", S_xx)
end

if ~isfile("data/medium_controls_main.jld") || force_generate
    data = CSV.read(datadep"VarianceComponentsHDFE/medium_controls_main.csv"; header=true)
    Xcontrols, S_xx = compute_X_Controls(data)
    save("data/medium_controls_main.jld", "Xcontrols", Xcontrols, "S_xx", S_xx)
end


# TODO: This is throwing a Killed: 9 error
# if ~isfile("data/full_main.jld") || force_generate
#    data = CSV.read(datadep"VarianceComponentsHDFE/full_main.csv"; header=true)
#    X_Laplacian, X_GroundedLaplacian, S_xx, X̃, R_p, R_b, A_d, A_f = compute_X_No_Controls(data)
#    save("data/full_main.jld", "X_Laplacian", X_Laplacian, "X_GroundedLaplacian", X_GroundedLaplacian, "S_xx", S_xx, "X_tilde", X̃, "R_p", R_p, "R_b", R_b, "A_d", A_d, "A_f", A_f)
# end
