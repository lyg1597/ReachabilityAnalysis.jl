import numpy as np
import scipy
from scipy.integrate import odeint

# quadrotor physical constants
g = 9.81; d0 = 10; d1 = 8; n0 = 10; kT = 0.91

# non-linear dynamics
def f(x, u):
    x, vx, theta_x, omega_x, y, vy, theta_y, omega_y, z, vz = x.reshape(-1).tolist()
    az, ax, ay = u.reshape(-1).tolist()
    dot_x = np.array([
     vx,
     g * np.tan(theta_x),
     -d1 * theta_x + omega_x,
     -d0 * theta_x + n0 * ax,
     vy,
     g * np.tan(theta_y),
     -d1 * theta_y + omega_y,
     -d0 * theta_y + n0 * ay,
     vz,
     kT * az - g])
    return dot_x

# linearization
# The state variables are x, y, z, vx, vy, vz, theta_x, theta_y, omega_x, omega_y
A = np.zeros([10, 10])
A[0, 3] = 1.
A[1, 4] = 1.
A[2, 5] = 1.
A[3, 6] = g
A[4, 7] = g
A[6, 6] = -d1
A[6, 8] = 1
A[7, 7] = -d1
A[7, 9] = 1
A[8, 6] = -d0
A[9, 7] = -d0

B = np.zeros([10, 3])
B[5, 0] = kT
B[8, 1] = n0
B[9, 2] = n0

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return np.asarray(K), np.asarray(X), np.asarray(eigVals)

####################### solve LQR #######################
n = A.shape[0]
m = B.shape[1]
Q = np.eye(n)
Q[0, 0] = 10.
Q[1, 1] = 10.
Q[2, 2] = 10.
R = np.diag([1., 1., 1.])
K, _, _ = lqr(A, B, Q, R)

####################### The controller ######################
def u(x, goal):
    goal = np.array(goal)
    return K.dot([goal[0],0,0,0, goal[1],0,0,0, goal[2],0] - x) + [0, 0, g / kT]

######################## The closed_loop system #######################
def cl_nonlinear(x, t, goal):
    x = np.array(x)
    dot_x = f(x, u(x, goal))
    return dot_x

# simulate
def simulate(x, goal, dt):
    curr_position = np.array(x)[[0, 4, 8]]
    error = goal - curr_position
    distance = np.sqrt((error**2).sum())
    if distance > 1:
        goal = curr_position + error / distance
    return odeint(cl_nonlinear, x, [0, dt], args=(goal,))[-1]
if __name__ == "__main__":
    goal = [5,5,5]
    x = [0,0,0,0,0,0,0,0,0,0]
    print(K)
    for i in range(10):
        x = simulate(x, goal, 0.1)
        print(x)