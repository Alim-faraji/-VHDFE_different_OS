using LinearAlgebra
using Random 
using DataFrames
using Statistics
using SparseArrays
using IterativeSolvers
using LightGraphs
using VectorizedRoutines
#using CSVFiles
using DataFramesMeta
using CSV
#To install Laplacians you need to put in Julia's Prompt "add Laplacians#master"
#For some reason Pkg.add() doesn't work with this.
#Pkg.add("MATLAB")
#using Laplacians
#using Pkg
#include(string(Pkg.dir("Laplacians") , "/src/matlabSolvers.jl"))
using Test
using Parameters

#Full Network
const y_full = [0.146297; 0.29686 ;  0.54344; 0.432677 ; 0.464866 ; 0.730622 ; 0.619239; 0.753429; 0.0863208; 0.372084 ;  0.958089]
const id_full = [1; 1; 2;2 ; 3; 4;4 ;5;5 ;6;6]
const firmid_full = [1;2;1;1;1;1;2;2;2;2;3]
const obs_id_full = collect(1:11)

[obs_id_full id_full firmid_full  y_full]

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
@testset "matrices" begin
    @test find_connected_set(y_full,id_full,firmid_full; verbose=true) == (obs_id = obs_id_full , y = y_full , id = id_full, firmid = firmid_full )

    @test prunning_connected_set(y_full,id_full,firmid_full, obs_id_full; verbose=true) == (obs_id = obs_id_p , y = y_p , id = id_p, firmid = firmid_p )

    @test drop_single_obs(y_p,id_p,firmid_p,obs_id_p) == (obs_id = obs_id_lo , y = y_lo , id = id_lo, firmid = firmid_lo ) 
    @test drop_single_obs(y_p,id_p,firmid_p,obs_id_p) == (obs_id = obs_id_lo1 , y = y_lo , id = id_lo, firmid = firmid_lo )

    @test index_constr(obs_id_lo , id_lo, match_id) == [obs_id_lo  obs_id_lo  [1;2;3;3;4;5;6;6]   obs_id_lo]
    @test index_constr(match_id , id_lo, match_id) == [[1;2;3;3;4;5;6;7;7;8]  [1;2;3;4;4;5;6;7;8;8]  [1;2;3;3;3;4;5;6;6;6]   [1;2;3;3;3;4;5;6;6;6]  ]
end

X =  sparse([1,2,3,4,5,6,7,8,1,3,4,5,2,6,7,8], [1,1,2,2,3,3,4,4,5,5,5,5,6,6,6,6],[1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1])
Xprime = sparse([1,2,3,4,5,6,7,8,1,3,4,5], [1,1,2,2,3,3,4,4,5,5,5,5],[1,1,1,1,1,1,1,1,-1,-1,-1,-1])

settings_1 = Settings()
settings_2 = Settings(lls_algorithm = DirectLLS())
settings_3 = Settings(lls_algorithm = CMGPreconditionedLLS())
settings_4 = Settings(leverage_algorithm = JLAAlgorithm())
settings_5 = Settings(leverage_algorithm = JLAAlgorithm(), lls_algorithm = DirectLLS())


#Direct Method
@test lss(settings_1.lls_algorithm, X, X[1,:], settings_1)  ≈ [ 0.5166666666666669 ; -0.23333333333333325 ; 0.01666666666666683; 0.2666666666666666; -0.23333333333333314; 0.26666666666666666]

#AMGPreconditionedLLS
@test lss(settings_2.lls_algorithm, Xprime, Xprime[1,:], settings_2) ≈ [0.25; -0.5; -0.25; 0.0; -0.5]
@test_throws SingularException(6) lss(settings_2.lls_algorithm, X, X[1,:], settings_2) 

#CMGPreconditionedLLS
@test lss(settings_3.lls_algorithm, Xprime, Xprime[1,:], settings_3)  ≈ [0.25; -0.5; -0.25; 0.0; -0.5] 
@test lss(settings_3.lls_algorithm, X, X[1,:], settings_3)   ≈ [0.25; -0.5; -0.25; 0.0; -0.5; 0.0]

elist =  [obs_id_lo  obs_id_lo  [1;2;3;3;4;5;6;6]   obs_id_lo]
NT = 8
N = 4
J = 2
K = 0
id_movers = [1;1;3;3]
firmid_movers = [1;2;1;2]
Tinv = [0.5; 0.5; 0.5; 0.5; 0.5; 0.5; 0.5; 0.5 ]
elist_JLL = [id_movers N.+firmid_movers id_movers N.+firmid_movers]
M = 4


Xright = sparse([1;2;3;4;1;3;2;4], [1;1;3;3;5;5;6;6], [1.0;1.0;1.0;1.0;-1.0;-1.0;-1.0;-1.0],4,6)
Xfe = sparse([1;3;4;5;2;6;7;8], [5;5;5;5;6;6;6;6],[-1.0; -1.0; -1.0;-1.0;-1.0;-1.0;-1.0;-1.0],8,6)
Xpe = sparse([7;8], [1;1], [1.0;1.0], 8,3)
elist_1 = [1;1;3;3]
elist_2 = [5;6;5;6]
elist_3 = [1;1;3;3]
elist_4 = [5;6;5;6]


@test initialize_auxiliary_variables(settings_1.leverage_algorithm, X, elist_JLL,M, NT, N, J , K, settings_1)  == (Xright = Xright, Xleft =nothing)
@test initialize_auxiliary_variables(settings_4.leverage_algorithm, X, elist_JLL, M,NT, N, J , K, settings_4)  == (Xfe = Xfe, Xpe =Xpe, elist_1=elist_1,elist_2=elist_2, elist_3=elist_3,elist_4=elist_4)

# FIXIT: the following lines don't work
# This doesn't work but we should have something like :
# @test compute_leverages(settings_1.leverage_algorithm, X, AUX1.Xright[1,:], M, NT, N, J , settings_1).Pii ≈ [ 0.75, 0.108888888888   ]
#Random.seed!(1234)
#@test compute_leverages(settings_4.leverage_algorithm, X, AUX2, NT, N, J , settings_4).Pii ≈  [0.00166667, 0.00166667, 0.00166667, 0.00166667]

#@test eff_res(settings_1.leverage_algorithm, X,id_lo,firmid_lo, match_id, 0, settings_1) == something 
#@test eff_res(settings_4.leverage_algorithm, X,id_lo,firmid_lo, match_id, 0, settings_4) ==something 

