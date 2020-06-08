# VarianceComponentsHDFE

[![Build Status](https://github.com/jlperla/VarianceComponentsHDFE.jl/workflows/CI/badge.svg)](https://github.com/jlperla/VarianceComponentsHDFE.jl/actions)
[![Coverage](https://codecov.io/gh/jlperla/VarianceComponentsHDFE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jlperla/VarianceComponentsHDFE.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jlperla.github.io/VarianceComponentsHDFE.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jlperla.github.io/VarianceComponentsHDFE.jl/dev)


- [Rough Notes](https://github.com/jlperla/VarianceComponentsHDFE.jl/blob/gh_actions_builds/rough_notes.pdf)

## Code Style
- Use the unicode and/or math symbol matching the algebra whenever possible
- Follow https://github.com/jrevels/YASGuide where possible
    - https://github.com/QuantEcon/lecture-source-jl/blob/master/style.md also is useful, but is intended more for "scripting" code rather than package code.
- The `.JuliaFormatter.toml` file has settings for the automatic formatting of the code
  - To use it in Atom/Juno, use `<Ctrl+P>` then type in `format` and you will find it.
  - Before formatting code, consider ensureing the unit tests pass.  I haven't seen bugs with the formatting yet, but there may be some.
- It is essential to force the use of `LF` for anyone using Windows on any computer, to do this
  - Git config in https://julia.quantecon.org/more_julia/version_control.html#Setup
  - Atom settings in https://julia.quantecon.org/more_julia/tools_editors.html#Installing-Atom
  - VSCode settings (i.e. files.eof) in https://github.com/ubcecon/tutorials/blob/master/vscode.md#general-packages-and-setup
- The style in julia is to have minimal headers and separators in files.  So no big ugly comment headers separating sections
  - If you want to separate sections, start a comment with `## ` and it will have a visual separation in Atom.
