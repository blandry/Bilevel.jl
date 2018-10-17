module BilevelTrajOpt

export 

using StaticArrays
using ForwardDiff
using Ipopt
using Rotations
using RigidBodyDynamics
using RigidBodyDynamics.Contact
using RigidBodyTreeInspector
using DrakeVisualizer

include("contact.jl")

end # module
