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

using Test
using StaticArrays
using LinearAlgebra
using RigidBodyDynamics
using RigidBodyDynamics: HalfSpace3D, separation
using Rotations
using CoordinateTransformations: transform_deriv
using MechanismGeometries
using GeometryTypes: HyperSphere, origin, radius, HyperRectangle
using ForwardDiff
using DiffResults
using Ipopt
using Snopt
using Compat

include("auglag.jl")
include("ip.jl")
include("environments.jl")
include("contact.jl")
include("simulation.jl")
include("simulation_snopt.jl")
include("svd.jl")

end # module
