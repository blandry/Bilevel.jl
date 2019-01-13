module BilevelTrajOpt

export HalfSpace,
       Obstacle,
       Environment,
       planar_obstacle,
       contact_basis,
       parse_contacts,
       auglag_solve,
       ip_solve,
       simulate,
       simulate_snopt,
       get_sim_data,
       separation,
       solve_implicit_contact_τ,
       parse_contacts,
       svd

using StaticArrays
using Ipopt
using Base.Test
using RigidBodyDynamics
using RigidBodyDynamics: HalfSpace3D, separation
using Rotations
using CoordinateTransformations: transform_deriv
using ForwardDiff
using ReverseDiff
using MechanismGeometries
using GeometryTypes: HyperSphere, origin, radius, HyperRectangle
using Compat
using Snopt

include("auglag.jl")
include("ip.jl")
include("environments.jl")
include("contact.jl")
include("simulation.jl")
include("simulation_snopt.jl")
include("svd.jl")

end # module
