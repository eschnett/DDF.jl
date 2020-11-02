# Run `julia --project=@. --color=yes make.jl` to build the
# documentation. The `Documenter` must be available (i.e. installed
# globally) for this to work.

using DDF
using Documenter

makedocs(sitename="DDF"; format=Documenter.HTML(prettyurls=false))
