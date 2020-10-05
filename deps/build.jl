using PackageCompiler
create_app(".", "deps/VarianceComponentsHDFELinux", force=true, precompile_statements_file="deps/precompilation_statements.jl")
