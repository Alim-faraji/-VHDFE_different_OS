using VarianceComponentsHDFE
using Test

include("prepare_test_data.jl")

@testset "VarianceComponentsHDFE.jl" begin
    include("solver.jl")
    include("test_matrices.jl")
end
