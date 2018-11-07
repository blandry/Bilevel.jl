module BilevelTrajOpt

export Obstacle,
       Environment,
       planar_obstacle,
       contact_basis,
       auglag_solve,
       ip_solve,
       f_contact,
       h_contact,
       g_contact,
       τ_contact_wrenches,
       update_constraints_implicit_contact,
       simulate,
       simulate_implicit


using StaticArrays
using Ipopt
using OSQP
using Base.Test
using RigidBodyDynamics
using RigidBodyDynamics: HalfSpace3D, separation
using Rotations
using CoordinateTransformations: transform_deriv
using ForwardDiff
using MechanismGeometries
using GeometryTypes: HyperSphere, origin, radius
using Compat

include("bilevel.jl")
include("contact.jl")
include("environments.jl")
include("simulation.jl")

end # module
