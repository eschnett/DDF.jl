# Run `julia --project=@. --color=yes make.jl` to build the
# documentation

using DDF
using Documenter

makedocs(sitename="DDF"; format=Documenter.HTML(prettyurls=false))
