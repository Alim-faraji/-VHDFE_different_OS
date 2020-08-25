# VarianceComponentsHDFE

[![Build Status](https://github.com/HighDimensionalEconLab/VarianceComponentsHDFE.jl/workflows/CI/badge.svg)](https://github.com/HighDimensionalEconLab/VarianceComponentsHDFE.jl/actions)
![LaTeX](https://github.com/HighDimensionalEconLab/VarianceComponentsHDFE.jl/workflows/LaTeX/badge.svg)
[![Coverage](https://codecov.io/gh/HighDimensionalEconLab/VarianceComponentsHDFE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HighDimensionalEconLab/VarianceComponentsHDFE.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HighDimensionalEconLab.github.io/VarianceComponentsHDFE.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HighDimensionalEconLab.github.io/VarianceComponentsHDFE.jl/dev)


- [Rough Notes](https://github.com/HighDimensionalEconLab/VarianceComponentsHDFE.jl/blob/gh_actions_builds/rough_notes.pdf)

## Code Style
- Use the unicode and/or math symbol matching the algebra whenever possible
- Follow https://github.com/jrevels/YASGuide where possible
    - https://github.com/QuantEcon/lecture-source-jl/blob/master/style.md also is useful, but is intended more for "scripting" code rather than package code.
- The `.JuliaFormatter.toml` file has settings for the automatic formatting of the code
  - To use it in Atom/Juno, use `<Ctrl+P>` then type in `format` and you will find it.
  - Before formatting code, consider ensureing the unit tests pass.  I haven't seen bugs with the formatting yet, but there may be some.
  - For vscode, install https://marketplace.visualstudio.com/items?itemName=singularitti.vscode-julia-formatter then in your    `settings.json` add in (following the instructions there)
    ```
      "[julia]": {
        "editor.defaultFormatter": "singularitti.vscode-julia-formatter"
    },
    ```
  - In vscode if you then go `ctrl-p` and type `format` you will be able to choose the singularitti one as your default julia editor
- It is essential to force the use of `LF` for anyone using Windows on any computer, to do this
  - Git config in https://julia.quantecon.org/more_julia/version_control.html#Setup
  - Atom settings in https://julia.quantecon.org/more_julia/tools_editors.html#Installing-Atom
  - VSCode settings (i.e. files.eof) in https://github.com/ubcecon/tutorials/blob/master/vscode.md#general-packages-and-setup
- The style in julia is to have minimal headers and separators in files.  So no big ugly comment headers separating sections
  - If you want to separate sections, start a comment with `## ` and it will have a visual separation in Atom.

  ## Compiling an Executable
  - To compile an executable of the package run the following Julia commands.
  ```
    using PackageCompiler
    compile_app("PathToPackageDir/VarianceComponentsHDFE", "PathToExecutableDir/VarianceComponentsHDFEExecutable)

  ```
  - To run the executable you can execute in the command line as follows:
  ```
    PathToExecutableDir/VarianceComponentsHDFEExecutable/bin/VarianceComponentsHDFE PathToCSVDataFile/data.csv --id=1 --firmid=2 --y=4 --algorithm=JLA --simulations=1000 --write_CSV --output_path=PathToCSVOutput/output.csv

  ```
  - The first argument is required and it is the path the the CSV file containing the data. The options `--id`, `--firmid`, `--y` indicate which columns of the CSV file contain the data for the worker IDs, firm IDs, and wages. `--algorithm` can be set to `Exact` or `JLA` and `--simulations` is the number of simulations in the JLA algorithm. `--write_CSV` is a flag that indicates the output will be written to a CSV file at `--output_path`. Additionally, you can run `PathToExecutableDir/VarianceComponentsHDFEExecutable/bin/VarianceComponentsHDFE --help` to see the arguments, options, and flags and their descriptions, and if applicable, default values.
