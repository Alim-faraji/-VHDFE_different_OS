import Pkg
Pkg.add("PackageCompiler")
using PackageCompiler
create_app(".","deps/VarianceComponentsHDFELinux",force=true)
