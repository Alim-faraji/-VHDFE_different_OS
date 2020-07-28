using LinearAlgebra, SparseArrays,Statistics,DataFrames,DataFramesMeta

#To install Laplacians you need to put in Julia's Prompt "add Laplacians#master"
#For some reason Pkg.add() doesn't work with this.
#Pkg.add("MATLAB")
#using Laplacians
#using Pkg
#include(string(Pkg.dir("Laplacians") , "/src/matlabSolvers.jl"))
using Test
using Parameters
using Random

#Full Network
const y_full = [0.146297; 0.29686 ;  0.54344; 0.432677 ; 0.464866 ; 0.730622 ; 0.619239; 0.753429; 0.0863208; 0.372084 ;  0.958089]
const id_full = [1; 1; 2;2 ; 3; 4;4 ;5;5 ;6;6]
const firmid_full = [1;2;1;1;1;1;2;2;2;2;3]
const obs_id_full = collect(1:11)

#Pruned Network
const y_p = [0.146297; 0.29686 ;  0.54344; 0.432677 ; 0.464866 ; 0.730622 ; 0.619239; 0.753429; 0.0863208]
const id_p = [1; 1; 2;2 ; 3; 4;4 ;5;5]
const firmid_p = [1;2;1;1;1;1;2;2;2]
const obs_id_p = collect(1:9)

#Drop-Single Network
const y_lo = [0.146297; 0.29686 ;  0.54344; 0.432677 ; 0.730622 ; 0.619239; 0.753429; 0.0863208]
const id_lo = [1; 1; 2;2 ; 3;3 ;4;4]
const firmid_lo = [1;2;1;1;1;2;2;2]
const obs_id_lo1 = [1;2;3;4;6;7;8;9]
const obs_id_lo = collect(1:8)

#Leave-Out Network Matchid
const match_id = [1; 2 ; 3;3 ; 4; 5; 6; 6]
@testset "PruneNetwork" begin
    @test find_connected_set(y_full,id_full,firmid_full; verbose=true) == (obs_id = obs_id_full , y = y_full , id = id_full, firmid = firmid_full )

    @test prunning_connected_set(y_full,id_full,firmid_full, obs_id_full; verbose=true) == (obs_id = obs_id_p , y = y_p , id = id_p, firmid = firmid_p )

    @test drop_single_obs(y_p,id_p,firmid_p,obs_id_p) == (obs_id = obs_id_lo1 , y = y_lo , id = id_lo, firmid = firmid_lo )

end

@testset "Movers & Matches" begin 
    @test  compute_matchid(firmid_lo,id_lo) == [1; 2 ; 3;3 ; 4; 5; 6; 6]
    @test  compute_movers(id_lo, firmid_lo) == (movers = Bool[1,1,0,0,1,1,0,0], T= [2, 2, 2, 2, 2, 2, 2, 2])
end


Xtest = sparse([1;2;3;4;5;6;7;8;1;3;4;5],[1;1;2;2;3;3;4;4;5;5;5;5],[1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0;-1.0;-1.0;-1.0;-1.0])
xxtest = Xtest'*Xtest
compute_sol = approxcholSolver(xxtest;verbose=true)
settings_exact = Settings(leverage_algorithm = ExactAlgorithm(), person_effects=true, cov_effects=true)
settings_JLA = Settings(leverage_algorithm = JLAAlgorithm(), person_effects=true, cov_effects=true)

@testset "Clustering" begin
    @test check_clustering(id_lo).nnz_2 == 12

    @test check_clustering(match_id).nnz_2 ==10

    @test index_constr(obs_id_lo,id_lo, obs_id_lo) == [obs_id_lo  obs_id_lo  obs_id_lo   obs_id_lo]

    @test index_constr(obs_id_lo,id_lo, match_id) == [obs_id_lo  obs_id_lo  [1;2;3;3;4;5;6;6]   obs_id_lo]

end


Random.seed!(1234)
rademacher1 = rand(1,8)  .> 0.5
rademacher1 = rademacher1 - (rademacher1 .== 0)

@testset "Solvers" begin 
    @test compute_sol( [Xtest[1,:]...] ;verbose=true) ≈ [ 0.25, -0.5, -0.25, 0.0, -0.5] 
    @test compute_sol([rademacher1*Xtest...];verbose=true) ≈ [ 1.0, 0.0, 1.0, -1.0, 0.0] 
end

@testset "LambdaMatrices" begin 
    @test    do_Pii(Xtest,obs_id_lo) == sparse( [1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8], [0.75,0.75,0.5,0.5,0.75,0.75,0.5,0.5] )
    #@test    eff_res(settings_exact.leverage_algorithm, X,id,firmid,match, K, settings_exact).Lambda_P == sparse( [1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8], [0.75,0.75,0.5,0.5,0.75,0.75,0.5,0.5] )
    #@test    eff_res(settings_JLA.leverage_algorithm, X,id,firmid,match, K, settings_JLA).Lambda_P ≈ sparse( [1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8], [0.75,0.75,0.5,0.5,0.75,0.75,0.5,0.5] )
end

@testset "LeaveOut" begin 
    #  @test  leave_out_estimation(y,id,firmid,nothing, settings_exact) ≈ [-0.0042 , 0.0019, 0.0037]
    #  @test  leave_out_estimation(y,id,firmid,nothing, settings_JLA) ≈ [-0.0042 , 0.0019, 0.0037]
end

