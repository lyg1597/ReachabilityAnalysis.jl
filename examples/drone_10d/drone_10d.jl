# # Quadrotor
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/Quadrotor.ipynb)
#
#md # !!! note "Overview"
#md #     System type: polynomial continuous system\
#md #     State dimension: 2\
#md #     Application domain: Chemical kinetics
#
# ## Model description
#
# We study the dynamics of a quadrotor as derived in [xxx]. Let us first introduce
# the variables required to describe the model: the inertial (north) position
# ``x_1``, the inertial (east) position ``x_2``, the altitude ``x_3``, the
# longitudinal velocity ``x_4``, the lateral velocity ``x_5``, the vertical
# velocity ``x_6``, the roll angle ``x_7``, the pitch angle ``x_8``, the yaw
# angle ``x_9``, the roll rate ``x_{10}``, the pitch rate ``x_{11}``, and the
# yaw rate ``x_{12}``. We further  require the following parameters: gravity
# constant ``g = 9.81`` [m/s``^2``], radius of center mass ``R = 0.1`` [m],
# distance of motors to center mass ``l = 0.5`` [m], motor mass
# ``M_{rotor} = 0.1`` [kg], center mass ``M = 1`` [kg], and total mass
# ``m = M + 4M_{rotor}``.
#
# From the above parameters we can compute the moments of inertia as
#
# ```math
# \begin{array}{lcll}
# J_x & = & \frac{2}{5}, M, R^2 + 2, l^2, M_{rotor}, \\
# J_y & = & J_x, \\
# J_z & = & \frac{2}{5}, M, R^2 + 4, l^2, M_{rotor}.
# \end{array}
# ```
#
# Finally, we can write the set of ordinary differential equations for the
# quadrotor according to [xxx]:
# ```math
# \left\{
# \begin{array}{lcl}
# \dot{x}_1 & = & \cos(x_8)\cos(x_9)x_4 + \Big(\sin(x_7)\sin(x_8)\cos(x_9) - \cos(x_7)\sin(x_9)\Big)x_5 \\
# & & + \Big(\cos(x_7)\sin(x_8)\cos(x_9) + \sin(x_7)\sin(x_9)\Big)x_6 \\
# \dot{x}_2 & = & \cos(x_8)\sin(x_9)x_4 + \Big(\sin(x_7)\sin(x_8)\sin(x_9) + \cos(x_7)\cos(x_9)\Big)x_5 \\
# & & + \Big(\cos(x_7)\sin(x_8)\sin(x_9) - \sin(x_7)\cos(x_9)\Big)x_6 \\
# \dot{x}_3 & = & \sin(x_8)x_4 - \sin(x_7)\cos(x_8)x_5 - \cos(x_7)\cos(x_8)x_6 \\
# \dot{x}_4 & = & x_{12}x_5 - x_{11}x_6 - g\sin(x_8) \\
# \dot{x}_5 & = & x_{10}x_6 - x_{12}x_4 + g\cos(x_8)\sin(x_7) \\
# \dot{x}_6 & = & x_{11}x_4 - x_{10}x_5 + g\cos(x_8)\cos(x_7) - \frac{F}{m} \\
# \dot{x}_7 & = & x_{10} + \sin(x_7)\tan(x_8)x_{11} + \cos(x_7)\tan(x_8)x_{12} \\
# \dot{x}_8 & = & \cos(x_7)x_{11} - \sin(x_7)x_{12} \\
# \dot{x}_9 & = & \frac{\sin(x_7)}{\cos(x_8)}x_{11} + \frac{\cos(x_7)}{\cos(x_8)}x_{12} \\
# \dot{x}_{10} & = & \frac{J_y - J_z}{J_x}x_{11}x_{12} + \frac{1}{J_x}\tau_\phi \\
# \dot{x}_{11} & = & \frac{J_z - J_x}{J_y}x_{10}x_{12} + \frac{1}{J_y}\tau_\theta \\
# \dot{x}_{12} & = & \frac{J_x - J_y}{J_z}x_{10}x_{11} + \frac{1}{J_z}\tau_\psi
# \end{array}
# \right.
# ```
#
# To check interesting control specifications, we stabilize the quadrotor using
# simple PD controllers for height, roll, and pitch. The inputs to the
# controller are the desired values for height, roll, and pitch ``u_1``, ``u_2``,
# and ``u_3``, respectively. The equations of the controllers are:
# ```math
# \begin{array}{lcll}
# F & = & m \, g - 10(x_3 - u_1) + 3x_6 \; & (\text{height control}), \\
# \tau_\phi & = & -(x_7 - u_2) - x_{10} & (\text{roll control}), \\
# \tau_\theta & = & -(x_8 - u_3) - x_{11} & (\text{pitch control}).
# \end{array}
# ```

# We leave the heading uncontrolled so that we set ``\tau_\psi = 0``.


using ReachabilityAnalysis
using ReachabilityAnalysis: is_intersection_empty

# ## Reachability settings
#
#The task is to change the height from ``0``~[m] to ``1``~[m] within ``5``~[s].
# A goal region ``[0.98,1.02]`` of the height ``x_3`` has to be reached within
# ``5``~[s] and the height has to stay below ``1.4`` for all times. After
# ``1``~[s] the height should stay above ``0.9``~[m]. The initial value for the
# position and velocities (i.e., from ``x_1`` to ``x_6``) is uncertain and given
# by ``[-\Delta,\Delta]``~[m], with ``\Delta=0.4``. All other variables are
# initialized to ``0``. This preliminary analysis must be followed by a
# corresponding evolution for ``\Delta = 0.1`` and ``\Delta= 0.8`` while keeping
# all the settings the same. No goals are specified for these cases: the
# objective instead is to understand the scalability of each tool with fixed
# settings.

const Tspan = (0.0, 5.0)

@taylorize function drone_10d!(dx, x, params, t)
    # dx[1] = 1.4*x[3] - 0.9*x[1]
    # dx[2] = 2.5*x[5] - 1.5*x[2]
    # dx[3] = 0.6*x[7] - 0.8*(x[2]*x[3])
    # dx[4] = 2 - 1.3*(x[3]*x[4])
    # dx[5] = 0.7*x[1] - (x[4]*x[5])
    # dx[6] = 0.3*x[1] - 3.1*x[6]
    # dx[7] = 1.8*x[6] - 1.6*(x[2]*x[7])
    dx[0+1]=x[1+1];
    dx[1+1]=9.81*tan(x[2+1]);
    dx[2+1]=-8*x[2+1] + x[3+1];
    dx[3+1]=-31.6227766*x[0+1] - 45.1663196*x[1+1] - 52.5011267*x[2+1] - 13.6015533*x[3+1] + 8.81028908e-17*x[4+1] - 4.23825517e-16*x[5+1] - 6.54099623e-16*x[6+1] + 1.59750882e-16*x[7+1] - 5.67884552e-16*x[8+1] - 1.48006779e-15*x[9+1] + 158.113883;
    dx[4+1]=x[5+1];
    dx[5+1]=9.81*tan(x[6+1]);
    dx[6+1]=-8*x[6+1] + x[7+1];
    dx[7+1]=2.05880071e-15*x[0+1] + 2.64242752e-15*x[1+1] + 2.62071672*x[2+1] + 1.59750882e-16*x[3+1] - 10.0*x[4+1] - 18.322492*x[5+1] - 24.6496768*x[6+1] - 11.3709874*x[7+1] + 1.9321227e-15*x[8+1] + 2.67619048e-15*x[9+1] - 1.029400355e-14;
    dx[8+1]=x[9+1];
    dx[9+1]=-5.0652413175e-17*x[0+1] - 6.7108633774e-17*x[1+1] - 1.194713975e-16*x[2+1] - 1.2256441379e-17*x[3+1] + 9.306933363e-17*x[4+1] + 1.2985133525e-16*x[5+1] + 1.0396072232e-16*x[6+1] + 2.2161533394e-17*x[7+1] - 0.91*x[8+1] - 1.6272983727*x[9+1];

    return dx
end

function drone_10d()
    ## initial states
    X0 = Hyperrectangle(low=[-1,0,0,0,-1,0,0,0,-1,0], high=[1,0,0,0,1,0,0,0,1,0])

    ## initial-value problem
    prob = @ivp(x' = drone_10d!(x), dim: 10, x(0) ∈ X0)

    return prob
end

# ----------------------------------------
#  Case 1: smaller uncertainty
# ----------------------------------------
Wpos = 0.1
Wvel = 0.1
prob = drone_10d()
alg = TMJets(abstol=1e-7, orderT=5, orderQ=1, adaptive=false)

# Warm-up run
sol1 = solve(prob, tspan=Tspan, alg=alg);
solz1 = overapproximate(sol1, Zonotope);

using Plots

Plots.plot(solz1, vars=(0, 1), linecolor="blue", color=:blue, alpha=0.8,
    xlab="t", ylab="x")
savefig("./t-x.png")

# Plots.plot(solz1, vars=(0, 5), linecolor="blue", color=:blue, alpha=0.8,
#     xlab="t", ylab="y")
# savefig("./t-y.png")

# Plots.plot(solz1, vars=(0, 9), linecolor="blue", color=:blue, alpha=0.8,
#     xlab="t", ylab="z")
# savefig("./t-z.png")
