module BilevelTrajOpt

export contact_basis,
       contact_distance,
       contact_velocity

using StaticArrays
using ForwardDiff
using Rotations
using RigidBodyDynamics
using RigidBodyDynamics.Contact
using RigidBodyTreeInspector
using DrakeVisualizer
using JuMP
using Ipopt

include("contact.jl")

end # module
