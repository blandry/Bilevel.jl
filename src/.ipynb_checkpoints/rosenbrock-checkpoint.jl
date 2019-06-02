function solve_rosenbrock()
    function get_rosenbrock()
        function eval_obj(x::AbstractArray{T}) where T
            f = (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2

            f
        end

        function eval_cons(x::AbstractArray{T}) where T
            g = vcat(-5. .- x, x .- 5.)

            g
        end

        return generate_autodiff_solver_fn(eval_obj,eval_cons,[],1:4)
    end

    solver_fn = get_rosenbrock()

    x0 = [0.0; 0.0]
    options = Dict{String, Any}()
    options["Derivative option"] = 1
    options["Verify level"] = 1
    xopt, info = Bilevel.snopt(solver_fn, 0, 4, x0, options)    

    function get_barnesgrad()
        a1 = 75.196
        a3 = 0.12694
        a5 = 1.0345e-5
        a7 = 0.030234
        a9 = 3.5256e-5
        a11 = 0.25645
        a13 = 1.3514e-5
        a15 = -5.2375e-6
        a17 = 7.0e-10
        a19 = -1.6638e-6
        a21 = 0.0005
        a2 = -3.8112
        a4 = -2.0567e-3
        a6 = -6.8306
        a8 = -1.28134e-3
        a10 = -2.266e-7
        a12 = -3.4604e-3
        a14 = -28.106
        a16 = -6.3e-8
        a18 = 3.4054e-4
        a20 = -2.8673

        function eval_obj(x::AbstractArray{T}) where T
            x1 = x[1]
            x2 = x[2]
            y1 = x1*x2
            y2 = y1*x1
            y3 = x2^2
            y4 = x1^2

            f = a1 + a2*x1 + a3*y4 + a4*y4*x1 + a5*y4^2 +
                a6*x2 + a7*y1 + a8*x1*y1 + a9*y1*y4 + a10*y2*y4 +
                a11*y3 + a12*x2*y3 + a13*y3^2 + a14/(x2+1) +
                a15*y3*y4 + a16*y1*y4*x2 + a17*y1*y3*y4 + a18*x1*y3 +
                a19*y1*y3 + a20*exp(a21*y1)

            f
        end

        function eval_cons(x::AbstractArray{T}) where T
            x1 = x[1]
            x2 = x[2]
            y1 = x1*x2
            y2 = y1*x1
            y3 = x2^2
            y4 = x1^2

            g = zeros(T, 3)
            g[1] = 1 - y1/700.0
            g[2] = y4/25.0^2 - x2/5.0
            g[3] = (x1/500.0- 0.11) - (x2/50.0-1)^2

            g
        end

        fres = DiffResults.HessianResult(zeros(2))
        fcfg = ForwardDiff.HessianConfig(eval_obj, fres, zeros(2))
        gres = DiffResults.JacobianResult(zeros(3), zeros(2))
        gcfg = ForwardDiff.JacobianConfig(eval_cons, zeros(2))

        return Bilevel.generate_autodiff_solver_fn(eval_obj,fres,fcfg,eval_cons,gres,gcfg,[],1:3)
    end

    solver_fn = get_barnesgrad()

    x0 = [10., 10.]
    options = Dict{String, Any}()
    options["Derivative option"] = 1
    options["Verify level"] = 1
    xopt, info = Bilevel.snopt(solver_fn, 0, 3, x0, options)
    
    xopt
end
