__precompile__(false)

if Sys.islinux()
    const snoptlib = "/usr/local/lib/libsnopt_c.so"    
else
    const snoptlib = "/usr/local/lib/libsnopt_c.dylib"
end
const codes = Dict{Int64, String}()
codes[1] = "Finished successfully: optimality conditions satisfied"
codes[2] = "Finished successfully: feasible point found"
codes[3] = "Finished successfully: requested accuracy could not be achieved"
codes[11] = "The problem appears to be infeasible: infeasible linear constraints"
codes[12] = "The problem appears to be infeasible: infeasible linear equalities"
codes[13] = "The problem appears to be infeasible: nonlinear infeasibilities minimized"
codes[14] = "The problem appears to be infeasible: infeasibilities minimized"
codes[15] = "The problem appears to be infeasible: infeasible linear constraints in QP subproblem"
codes[21] = "The problem appears to be unbounded: unbounded objective"
codes[22] = "The problem appears to be unbounded: constraint violation limit reached"
codes[31] = "Resource limit error: iteration limit reached"
codes[32] = "Resource limit error: major iteration limit reached"
codes[33] = "Resource limit error: the superbasics limit is too small"
codes[41] =  "Terminated after numerical difficulties: current point cannot be improved"
codes[42] =  "Terminated after numerical difficulties: singular basis"
codes[43] =  "Terminated after numerical difficulties: cannot satisfy the general constraints"
codes[44] =  "Terminated after numerical difficulties: ill-conditioned null-space basis"
codes[51] =  "Error in the user-supplied functions: incorrect objective derivatives"
codes[52] =  "Error in the user-supplied functions: incorrect constraint derivatives"
codes[61] =  "Undefined user-supplied functions: undefined function at the first feasible point"
codes[62] =  "Undefined user-supplied functions: undefined function at the initial point"
codes[63] =  "Undefined user-supplied functions: unable to proceed into undefined region"
codes[71] =  "User requested termination: terminated during function evaluation "
codes[74] =  "User requested termination: terminated from monitor routine"
codes[81] =  "Insufficient storage allocated: work arrays must have at least 500 elements"
codes[82] =  "Insufficient storage allocated: not enough character storage"
codes[83] =  "Insufficient storage allocated: not enough integer storage"
codes[84] =  "Insufficient storage allocated: not enough real storage"
codes[91] =  "Input arguments out of range: invalid input argument"
codes[92] =  "Input arguments out of range: basis file dimensions do not match this problem"
codes[141] =  "System error: wrong number of basic variables"
codes[142] =  "System error: error in basis package"
const PRINTNUM = 18
const SUMNUM = 19

# callback function
function objcon_wrapper(status_::Ptr{Clong}, n::Clong, x_::Ptr{Cdouble},
    needf::Clong, nF::Clong, f_::Ptr{Cdouble}, needG::Clong, lenG::Clong,
    G_::Ptr{Cdouble}, cu::Ptr{Cchar}, lencu::Clong, iu::Ptr{Clong},
    leniu::Clong, ru_::Ptr{Cdouble}, lenru::Clong)

    status = unsafe_load(status_)

    # check if solution finished, no need to calculate more
    if status >= 2
        return
    end

    # unpack design variables
    x = zeros(n)
    for i = 1:n
        x[i] = unsafe_load(x_, i)
    end
    # x = unsafe_load(x_)  # TODO: test this

    # call function
    res = objcon(x)
    if length(res) == 3
        J, c, fail = res
        ceq = Float64[]
        gradprovided = false
    elseif length(res) == 4
        J, c, ceq, fail = res
        gradprovided = false
    elseif length(res) == 5
        J, c, gJ, gc, fail = res
        ceq = Float64[]
        gradprovided = true
    else
        J, c, ceq, gJ, gc, gceq, fail = res
        gradprovided = true
    end

    # copy obj and con values into C pointer
    unsafe_store!(f_, J, 1)
    for i = 2 : nF - length(ceq)
        unsafe_store!(f_, c[i-1], i)
    end
    if !isempty(ceq)
        for i = nF - length(ceq) + 1 : nF
            unsafe_store!(f_, ceq[i-length(c)-1], i)
        end
    end

    # gradients  TODO: separate gradient computation in interface?
    if needG > 0 && gradprovided

        for j = 1:n
            # gradients of f
            unsafe_store!(G_, gJ[j], j)
        end

        k = n+1
        for i = 2 : nF - length(ceq)
            for j = 1:n
                unsafe_store!(G_, gc[i-1, j], k)
                k += 1
            end
        end
        for i = nF - length(ceq) + 1 : nF
            for j = 1:n
                unsafe_store!(G_, gceq[i-length(c)-1, j], k)
                k += 1
            end
        end
    end


    # check if solutions fails
    if fail
        unsafe_store!(status_, -1, 1)
    end
end

# c wrapper to callback function
const usrfun = @cfunction(objcon_wrapper, Cvoid, (Ptr{Clong}, Ref{Clong}, Ptr{Cdouble},
    Ref{Clong}, Ref{Clong}, Ptr{Cdouble}, Ref{Clong}, Ref{Clong}, Ptr{Cdouble},
    Ptr{Cchar}, Ref{Clong}, Ptr{Clong}, Ref{Clong}, Ptr{Cdouble}, Ref{Clong}))

# main call to snopt
function snopt(fun, x0, lb, ub, options;
               printfile = "snopt-print.out", sumfile = "snopt-summary.out")

    # call function
    res = fun(x0)
    if (length(res) == 3) || (length(res) == 5)
        J, c, _ = res
        ceq = Float64[]
    else
        J, c, ceq, _ = res
    end

    # TODO: there is a probably a better way than to use a global
    global objcon = fun

    # TODO: set a timer

    # setup
    Start = 0  # cold start  # TODO: allow warm starts
    nF = 1 + length(c) + length(ceq)  # 1 objective + constraints
    n = length(x0)  # number of design variables
    ObjAdd = 0.0  # no constant term added to objective (user can add themselves if desired)
    ObjRow = 1  # objective is first thing returned, then constraints

    # linear constraints (none for now)
    iAfun = Clong[1]
    jAvar = Clong[1]
    A = [0.0]  # TODO: change later
    lenA = 1
    neA = 0

    # nonlinear constraints (assume dense jacobian for now)
    lenG = nF*n
    neG = lenG
    iGfun = Array{Clong}(undef, lenG)
    jGvar = Array{Clong}(undef, lenG)
    k = 1
    for i = 1:nF
        for j = 1:n
            iGfun[k] = i
            jGvar[k] = j
            k += 1
        end
    end

    # bound constriaints (no infinite bounds for now)
    xlow = lb
    xupp = ub
    Flow = -1e20*ones(nF)  # TODO: check Infinite Bound size
    Fupp = zeros(nF)  # TODO: currently c <= 0, but perhaps change

    if !isempty(ceq) #equality constraints
        Flow[nF - length(ceq) + 1 : nF] .= 0.0
    end

    # names
    Prob = "opt prob"  # problem name TODO: change later
    nxname = 1  # TODO: change later
    xnames = Array{UInt8}(undef, nxname, 8)
    # xnames = ["TODOTODO"]
    nFname = 1  # TODO: change later
    Fnames = Array{UInt8}(undef, nFname, 8)
    # Fnames = ["TODOTODO"]

    # starting info
    x = x0
    xstate = zeros(n)
    xmul = zeros(n)
    F = zeros(nF)
    Fstate = zeros(nF)
    Fmul = zeros(nF)
    # INFO = 0
    INFO = Clong[0]
    mincw = Clong[0]  # TODO: check that these are sufficient
    miniw = Clong[0]
    minrw = Clong[0]
    nS = Clong[0]
    nInf = Clong[0]
    sInf = Cdouble[0]
    lencu = 1
    # cu = Array{UInt8}(undef, lencu, 8)
    cu = zeros(UInt8, lencu, 8)
    iu = Clong[0]
    leniu = length(iu)
    ru = [0.0]
    lenru = length(ru)

    # open files for printing
    iprint = PRINTNUM
    isumm = SUMNUM
    printerr = Clong[0]
    sumerr = Clong[0]
    ccall( (:snopenappend_, snoptlib), Cvoid,
        (Ref{Clong}, Cstring, Ptr{Clong}, Clong),
        iprint, printfile, printerr, sizeof(printfile))
    ccall( (:snopenappend_, snoptlib), Cvoid,
        (Ref{Clong}, Cstring, Ptr{Clong}, Clong),
        isumm, sumfile, sumerr, sizeof(sumfile))
    if printerr[1] != 0
        println("failed to open print file")
    end
    if sumerr[1] != 0
        println("failed to open summary file")
    end

    # temporary working arrays

    ltmpcw = 500
    cw = Array{Cchar}(undef, ltmpcw*8)

    ltmpiw = 500
    iw = Array{Clong}(undef, ltmpiw)

    ltmprw = 500
    rw = Array{Cdouble}(undef, ltmprw)

    # --- initialize ----
    ccall( (:sninit_, snoptlib), Cvoid,
        (Ref{Clong}, Ref{Clong}, Ptr{Cchar}, Ref{Clong}, Ptr{Clong},
        Ref{Clong}, Ptr{Cdouble}, Ref{Clong}, Clong),
        iprint, isumm, cw, ltmpcw, iw,
        ltmpiw, rw, ltmprw, sizeof(cw))

    # --- set options ----
    errors = Clong[0]

    for key in keys(options)
        value = options[key]
        buffer = string(key, repeat(" ", 55-length(key)))  # buffer length is 55 so pad with space.

        if length(key) > 55
            println("warning: invalid option, too long")
            continue
        end

        errors[1] = 0

        if typeof(value) == String

            value = string(value, repeat(" ", 72-length(value)))

            ccall( (:snset_, snoptlib), Cvoid,
                (Cstring, Ref{Clong}, Ref{Clong}, Ptr{Clong},
                Ptr{Cchar}, Ref{Clong}, Ptr{Clong}, Ref{Clong}, Ptr{Cdouble}, Ref{Clong}, Clong, Clong),
                value, iprint, isumm, errors,
                cw, ltmpcw, iw, ltmpiw, rw, ltmprw, sizeof(buffer), sizeof(cw))

        elseif isinteger(value)

            ccall( (:snseti_, snoptlib), Cvoid,
                (Cstring, Ref{Clong}, Ref{Clong}, Ref{Clong}, Ptr{Clong},
                Ptr{Cchar}, Ref{Clong}, Ptr{Clong}, Ref{Clong}, Ptr{Cdouble}, Ref{Clong}, Clong, Clong),
                buffer, value, iprint, isumm, errors,
                cw, ltmpcw, iw, ltmpiw, rw, ltmprw, sizeof(buffer), sizeof(cw))

        elseif isreal(value)

            ccall( (:snsetr_, snoptlib), Cvoid,
                (Cstring, Ref{Cdouble}, Ref{Clong}, Ref{Clong}, Ptr{Clong},
                Ptr{Cchar}, Ref{Clong}, Ptr{Clong}, Ref{Clong}, Ptr{Cdouble}, Ref{Clong}, Clong, Clong),
                buffer, value, iprint, isumm, errors,
                cw, ltmpcw, iw, ltmpiw, rw, ltmprw, sizeof(buffer), sizeof(cw))
        end

        # println(errors[1])

    end

    # --- set memory requirements --- #
    ccall( (:snmema_, snoptlib), Cvoid,
        (Ptr{Clong}, Ref{Clong}, Ref{Clong}, Ref{Clong}, Ref{Clong}, Ref{Clong}, Ref{Clong},
        Ptr{Clong}, Ptr{Clong}, Ptr{Clong},
        Ptr{Cchar}, Ref{Clong}, Ptr{Clong}, Ref{Clong}, Ptr{Cdouble}, Ref{Clong}, Clong),
        INFO, nF, n, nxname, nFname,
        neA, neG,
        mincw, miniw, minrw,
        cw, ltmpcw, iw, ltmpiw, rw, ltmprw, sizeof(cw))

    # --- resize arrays to match memory requirements
    lencw = mincw
    resize!(cw,lencw[1]*8)
    leniw = miniw
    resize!(iw,leniw[1])
    lenrw = minrw
    resize!(rw,lenrw[1])

    memkey = ("Total character workspace", "Total integer   workspace",
        "Total real      workspace")
    memvalue = (lencw,leniw,lenrw)
    for (key,value) in zip(memkey,memvalue)
        buffer = string(key, repeat(" ", 55-length(key)))  # buffer length is 55 so pad with space.
        errors[1] = 0
        ccall( (:snseti_, snoptlib), Cvoid,
            (Cstring, Ref{Clong}, Ref{Clong}, Ref{Clong}, Ptr{Clong},
            Ptr{Cchar}, Ref{Clong}, Ptr{Clong}, Ref{Clong}, Ptr{Cdouble}, Ref{Clong}, Clong, Clong),
            buffer, value, iprint, isumm, errors,
            cw, ltmpcw, iw, ltmpiw, rw, ltmprw, sizeof(buffer), sizeof(cw))
    end

    # --- call snopta ----
    ccall( (:snopta_, snoptlib), Cvoid,
        (Ref{Clong}, Ref{Clong}, Ref{Clong}, Ref{Clong}, Ref{Clong},Ref{Cdouble},
        Ref{Clong}, Cstring, Ptr{Cvoid}, Ptr{Clong}, Ptr{Clong}, Ref{Clong},
        Ref{Clong}, Ptr{Cdouble}, Ptr{Clong}, Ptr{Clong}, Ref{Clong}, Ref{Clong},
        Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cchar}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cchar},
        Ptr{Cdouble}, Ptr{Clong}, Ptr{Cdouble},
        Ptr{Cdouble}, Ptr{Clong},
        Ptr{Cdouble}, Ptr{Clong}, Ptr{Clong}, Ptr{Clong}, Ptr{Clong}, Ptr{Clong},
        Ptr{Clong}, Ptr{Cdouble}, Ptr{Cchar}, Ref{Clong}, Ptr{Clong}, Ref{Clong},
        Ptr{Cdouble}, Ref{Clong}, Ptr{Cchar}, Ref{Clong}, Ptr{Clong}, Ref{Clong},
        Ptr{Cdouble}, Ref{Clong},
        Clong, Clong, Clong, Clong, Clong),
        Start, nF, n, nxname, nFname, ObjAdd,
        ObjRow, Prob, usrfun, iAfun, jAvar, lenA,
        neA, A, iGfun, jGvar, lenG, neG,
        xlow, xupp, xnames, Flow, Fupp, Fnames,
        x, xstate, xmul, F, Fstate,
        Fmul, INFO, mincw, miniw, minrw, nS,
        nInf, sInf, cu, lencu, iu, leniu,
        ru, lenru, cw, lencw, iw, leniw,
        rw, lenrw,
        sizeof(Prob),sizeof(xnames),sizeof(Fnames),sizeof(cu),sizeof(cw))

    # println("done")

    # close output files
    ccall( (:snclose_, snoptlib), Cvoid,
        (Ref{Clong},),
        iprint)
    ccall( (:snclose_, snoptlib), Cvoid,
        (Ref{Clong},),
        isumm)

    return x, F[1], codes[INFO[1]]  # xstar, fstar, info

end