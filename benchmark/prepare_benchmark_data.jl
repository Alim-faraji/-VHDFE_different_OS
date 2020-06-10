using SparseArrays, LinearAlgebra
using JLD, CSV, DataFrames, DataFramesMeta

# add in code to generate .jld, .csv, etc. in the benchmark/data directory
# This should only generate if the file doesn't exist, or if this `force_generate = true`
force_generate = false

## Medium-Sized Network Generator
if isfile("data/test.jld") && ~force_generate
    data = load("data/test.jld")
    Xmedium_Laplacian = data["Xmedium_Laplacian"]
    Xmedium_GroundedLaplacian = data["Xmedium_GroundedLaplacian"]
else
    data = CSV.read("../data/test.csv"; header=false)
    data = DataFrame(id = data[:,1], firmid = data[:,2], year = data[:,3], y = data[:,4] )
    sort!(data, (:id, :year))

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

    #Worker Dummies
    D=sparse(collect(1:NT),id,1);

    #Firm Dummies
    F=sparse(collect(1:NT),firmid,1);

    # N+J x N+J-1 restriction matrix
    S= sparse(1.0I, J-1, J-1);
    S=vcat(S,sparse(-zeros(1,J-1)));

    Xmedium_Laplacian = hcat(D, -F)
    Xmedium_GroundedLaplacian = hcat(D, -F*S)

    # Save the matrices
    save("data/test.jld", "Xmedium_Laplacian", Xmedium_Laplacian, "Xmedium_GroundedLaplacian", Xmedium_GroundedLaplacian)
end