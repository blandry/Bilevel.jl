module BilevelTrajOpt

__precompile__(false)

export HalfSpace,
       Obstacle,
       Environment,
       planar_obstacle,
       contact_basis,
       parse_contacts,
       separation,
       svd,
       auglag_solve,
       ip_solve,
       snopt,
       solve_implicit_contact_τ,
       get_sim_data,
       simulate_ipopt,
       simulate_snopt,
       get_traj_data,
       add_state_eq!,
       add_fn_ineq!,
       add_fn_obj!,
       trajopt_snopt

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
using Compat

include("svd.jl")
include("auglag.jl")
include("ip.jl")
include("snopt.jl")
include("environments.jl")
include("contact.jl")
include("simulation.jl")
include("simulation_ipopt.jl")
include("simulation_snopt.jl")
include("trajopt.jl")
include("trajopt_snopt.jl")

end # module
