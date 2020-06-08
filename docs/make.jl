using VarianceComponentsHDFE
using Documenter

makedocs(;
    modules=[VarianceComponentsHDFE],
    authors="Various Collaborators",
    repo="https://github.com/jlperla/VarianceComponentsHDFE.jl/blob/{commit}{path}#L{line}",
    sitename="VarianceComponentsHDFE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jlperla.github.io/VarianceComponentsHDFE.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jlperla/VarianceComponentsHDFE.jl",
)
