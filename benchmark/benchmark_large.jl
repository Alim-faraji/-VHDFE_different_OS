using VarianceComponentsHDFE
using Random, BenchmarkTools, Test
using JLD, SparseArrays, LinearAlgebra
using AlgebraicMultigrid, IterativeSolvers
using Laplacians

#if (BLAS.vendor() == :openblas64 && !haskey(ENV, "OPENBLAS_NUM_THREADS"))
#    blas_num_threads = min(4, Int64(round(Sys.CPU_THREADS / 2)))  # or lower?
#    println("Setting BLAS threads = $blas_num_threads")
#    BLAS.set_num_threads(blas_num_threads)
#end

pkg_dir = pkgdir(VarianceComponentsHDFE)

# Obtain environment variables
use_matlab_CMG = get(ENV, "USE_MATLAB_CMG", "false") == "true" ? true : false
run_large_benchmark = get(ENV, "VCHDFE_LARGE_BENCHMARK", "false")  == "true" ? true : false
num_rhs_large = try
        parse(Int, get(ENV, "BENCHMARK_NUM_RHS_LARGE", "2"))
    catch
        2
    end

if use_matlab_CMG
    using  Pkg
    include(string(Pkg.dir("Laplacians") , "/src/matlabSolvers.jl"))
end


# Benchmarks for the no controls case in the JL algorithm
const SUITE = BenchmarkGroup()

# Load the data
include("prepare_benchmark_data.jl")
large_data = load(pkg_dir*"/benchmark/data/large_main.jld")
X = large_data["X_GroundedLaplacian"]
S_xx = large_data["S_xx"]
S_xx_sparse = sparse(S_xx) # The CMG solver can't handle a symmetric S_xx
m,k = size(X)


#SDDM solver
sol_sddm = approxchol_sddm(S_xx_sparse, verbose=true)

SUITE["Large: SDDM Solver Build for S_xx_sparse"] = @benchmarkable approxchol_sddm($S_xx_sparse, verbose=true)

sol_KMPsddm = KMPSDDMSolver(S_xx_sparse, maxits=300; verbose=true)
SUITE["Large: KMPSDDM Solver Build for S_xx_sparse"] = KMPSDDMSolver($S_xx_sparse, maxits=300; verbose=true)

#Create Adjacency and LAP solver
A = copy(S_xx_sparse)
A[diagind(A)] = spzeros(size(A,1))
A = -1*A
sol_lap = approxchol_lap(A; verbose=true)
SUITE["Large: Lap Solver Build for Adjacency Matrix"] = @benchmarkable approxchol_lap($A; verbose=true)

sol_cglap = Laplacians.cgLapSolver(A,maxits=300;verbose=true)
SUITE["Large: cgLap Solver Build for Adjacency Matrix"] = @benchmarkable  Laplacians.cgLapSolver($A,maxits=300;verbose=true)

sol_KMPlap = KMPLapSolver(A,maxits=300;verbose=true)
SUITE["Large: KMPLap Solver Build for Adjacency Matrix"] = @benchmarkable  KMPLapSolver($A,maxits=300;verbose=true)

# Only a single RHS, so set p = 1
R_p = convert(Array{Float64,2}, bitrand(1,m))
rademacher!(R_p)

# Setup the JLA_RHS
JLA_RHS = (R_p*X)[1,:]
JLA_RHS_sparse =  SparseMatrixCSC{Float64,Int64}(sparse(JLA_RHS))

#Laplacians Solvers benchmark
SUITE["Large: S_xx_sparse SDDM solver JL RHS"] = @benchmarkable sol_sddm($JLA_RHS, verbose=true)
SUITE["Large: S_xx_sparse LAP solver JL RHS"] = @benchmarkable sol_lap($JLA_RHS, verbose=true)
SUITE["Large: S_xx_sparse cgLAP solver JL RHS"] = @benchmarkable sol_cglap($JLA_RHS, verbose=true)
SUITE["Large: S_xx_sparse KMPSDDM solver JL RHS"] = @benchmarkable sol_KMPsddm($JLA_RHS, verbose=true)
SUITE["Large: S_xx_sparse KMPLap solver JL RHS"] = @benchmarkable sol_KMPlap($JLA_RHS, verbose=true)

# Setup and benchmark the precondiioner
SUITE["Large: S_xx precondition: AMG ruge_stuben"] = @benchmarkable aspreconditioner(ruge_stuben($S_xx))
SUITE["Large: S_xx_sparse precondition: AMG ruge_stuben"] = @benchmarkable aspreconditioner(ruge_stuben($S_xx_sparse))

P = aspreconditioner(ruge_stuben(S_xx))
P_sparse = aspreconditioner(ruge_stuben(S_xx_sparse))

# AMG/CG Benchmarks
z = 0.1.*ones(length(JLA_RHS))
SUITE["Large: S_xx iterative solve: AMG/CG"] = @benchmarkable cg!($z, $S_xx, $JLA_RHS, Pl = $P , log=true, maxiter=300)
z = 0.1.*ones(length(JLA_RHS))
SUITE["Large: S_xx_sparse iterative solve: AMG/CG"] = @benchmarkable cg!($z, $S_xx_sparse, $JLA_RHS, Pl = $P , log=true, maxiter=300)

z = 0.1.*ones(length(JLA_RHS))
SUITE["Large: S_xx iterative solve RHS_sparse: AMG/CG"] = @benchmarkable cg!($z, $S_xx, $JLA_RHS_sparse, Pl = $P , log=true, maxiter=300)
z = 0.1.*ones(length(JLA_RHS))
SUITE["Large: S_xx_sparse iterative solve RHS_sparse: AMG/CG"] = @benchmarkable cg!($z, $S_xx_sparse, $JLA_RHS_sparse, Pl = $P , log=true, maxiter=300)



z = 0.1.*ones(length(JLA_RHS))
SUITE["Large: S_xx iterative solve: AMG/CG, P_sparse"] = @benchmarkable cg!($z, $S_xx, $JLA_RHS, Pl = $P_sparse, log=true, maxiter=300)
z = 0.1.*ones(length(JLA_RHS))
SUITE["Large: S_xx_sparse iterative solve: AMG/CG, P_sparse"] = @benchmarkable cg!($z, $S_xx_sparse, $JLA_RHS, Pl = $P_sparse, log=true, maxiter=300)

z = 0.1.*ones(length(JLA_RHS))
SUITE["Large: S_xx iterative solve RHS_sparse: AMG/CG, P_sparse"] = @benchmarkable cg!($z, $S_xx, $JLA_RHS_sparse, Pl = $P_sparse, log=true, maxiter=300)
z = 0.1.*ones(length(JLA_RHS))
SUITE["Large: S_xx_sparse iterative solve RHS_sparse: AMG/CG, P_sparse"] = @benchmarkable cg!($z, $S_xx_sparse, $JLA_RHS_sparse, Pl = $P_sparse, log=true, maxiter=300)

if use_matlab_CMG
    SUITE["Large: S_xx iterative solve: CMG"] = @benchmarkable matlabCmgSolver($S_xx_sparse, $JLA_RHS; tol=1e-6, maxits=300)
    SUITE["Large: S_xx iterative solve: CMG, sparse RHS"] = @benchmarkable matlabCmgSolver($S_xx_sparse, $JLA_RHS_sparse; tol=1e-6, maxits=300)
end

