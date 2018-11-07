module BilevelTrajOpt

export 

using StaticArrays
using ForwardDiff
using Ipopt
using OSQP

include("contact.jl")
include("simulate.jl")

end # module
