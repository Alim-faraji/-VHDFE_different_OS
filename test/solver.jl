using SparseArrays, LinearAlgebra

X =  sparse([1,2,3,4,5,6,7,8,1,3,4,5,2,6,7,8], [1,1,2,2,3,3,4,4,5,5,5,5,6,6,6,6],[1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1])
Xprime = sparse([1,2,3,4,5,6,7,8,1,3,4,5], [1,1,2,2,3,3,4,4,5,5,5,5],[1,1,1,1,1,1,1,1,-1,-1,-1,-1])

settings_default = Settings()
settings_direct = Settings(lls_algorithm = DirectLLS())
settings_CMG = Settings(lls_algorithm = CMGPreconditionedLLS())


@testset "solver" begin
    #Direct Method
    @test lss(settings_default.lls_algorithm, X, X[1,:], settings_default)  ≈ [ 0.5166666666666669 ; -0.23333333333333325 ; 0.01666666666666683; 0.2666666666666666; -0.23333333333333314; 0.26666666666666666]

    #AMGPreconditionedLLS
    @test lss(settings_direct.lls_algorithm, Xprime, Xprime[1,:], settings_direct) ≈ [0.25; -0.5; -0.25; 0.0; -0.5]
    @test_throws SingularException(6) lss(settings_direct.lls_algorithm, X, X[1,:], settings_direct) 

    #CMGPreconditionedLLS (using MATLAB; don't run them in CI)
    # @test_throws MATLAB.MEngineError("failed to get variable iter from MATLAB session") lss(settings_CMG.lls_algorithm, Xprime, Xprime[1,:], settings_CMG)  # ≈ [0.25; -0.5; -0.25; 0.0; -0.5]
    # @test_throws MATLAB.MEngineError("failed to get variable iter from MATLAB session") lss(settings_CMG.lls_algorithm, X, X[1,:], settings_CMG)  # ≈ [0.25; -0.5; -0.25; 0.0; -0.5; 0.0]
end