import numpy as np 
from scipy.integrate import odeint
import random
import matplotlib.pyplot as plt

g = 9.81; d0 = 10; d1 = 8; n0 = 10; kT = 0.91

K = np.array([
    [3.16227766e+00, 4.51663196e+00, 4.25011267e+00, 1.36015533e+00, -8.81028908e-18, 4.23825517e-17, 6.54099623e-17, -1.59750882e-17, 5.67884552e-17, 1.48006779e-16],
    [-2.05880071e-16, -2.64242752e-16, -2.62071672e-16, -1.59750882e-17, 1.00000000e+00, 1.83224920e+00, 1.46496768e+00, 1.13709874e+00, -1.93212270e-16, -2.67619048e-16],
    [5.56619925e-17, 7.37457514e-17, 1.31287250e-16, 1.34686169e-17, -1.02273993e-16, -1.42693775e-16, -1.14242552e-16, -2.43533334e-17, 1.00000000e+00, 1.78823997e+00]
])

goal = [5,0,0]

def f(x, t):
    x, vx, thetax, omegax, y, vy, thetay, omegay, z, vz = x.reshape(-1).tolist()
    goal0 = (x+(goal[0]-x)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z)))
    goal1 = (y+(goal[1]-y)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z))) 
    goal2 = (z+(goal[2]-z)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z)))
    u1 = (3.16227766e+00*(goal0-x) - 4.51663196e+00*vx - 4.25011267e+00*thetax - 1.36015533e+00*omegax + (-8.81028908e-18)*(goal1-y) - 4.23825517e-17*vy - 6.54099623e-17*thetay - (-1.59750882e-17)*omegay + 5.67884552e-17*(goal2-z) - 1.48006779e-16*vz)
    u2 = ((-2.05880071e-16)*(goal0-x) - (-2.64242752e-16)*vx - (-2.62071672e-1)*thetax - (-1.59750882e-17)*omegax + 1.00000000e+00*(goal1-y) - 1.83224920e+00*vy - 1.46496768e+00*thetay - 1.13709874e+00*omegay + (-1.93212270e-16)*(goal2-z) - (-2.67619048e-16)*vz)
    u3 = (5.56619925e-17*(goal0-x) - 7.37457514e-17*vx - 1.31287250e-16*thetax - 1.34686169e-17*omegax + (-1.02273993e-16)*(goal1-y) - (-1.42693775e-16)*vy - (-1.14242552e-16)*thetay - (-2.43533334e-17)*omegay + 1.00000000e+00*(goal2-z) - 1.78823997e+00*vz + 9.81/0.91)
    
    # u1 = (3.16227766e+00*(goal[0]-x) - 4.51663196e+00*vx - 4.25011267e+00*thetax - 1.36015533e+00*omegax + (-8.81028908e-18)*(goal[1]-y) - 4.23825517e-17*vy - 6.54099623e-17*thetay - (-1.59750882e-17)*omegay + 5.67884552e-17*(goal[2]-z) - 1.48006779e-16*vz)
    # u2 = ((-2.05880071e-16)*(goal[0]-x) - (-2.64242752e-16)*vx - (-2.62071672e-1)*thetax - (-1.59750882e-17)*omegax + 1.00000000e+00*(goal[1]-y) - 1.83224920e+00*vy - 1.46496768e+00*thetay - 1.13709874e+00*omegay + (-1.93212270e-16)*(goal[2]-z) - (-2.67619048e-16)*vz)
    # u3 = (5.56619925e-17*(goal[0]-x) - 7.37457514e-17*vx - 1.31287250e-16*thetax - 1.34686169e-17*omegax + (-1.02273993e-16)*(goal[1]-y) - (-1.42693775e-16)*vy - (-1.14242552e-16)*thetay - (-2.43533334e-17)*omegay + 1.00000000e+00*(goal[2]-z) - 1.78823997e+00*vz + 9.81/0.91)
    
    ax, ay, az = u1, u2, u3
    # dot_x = np.array([
    #     vx,
    #     9.81 * np.tan(thetax),
    #     -8 * thetax + omegax,
    #     -10 * thetax + 10 * ax,
    #     vy,
    #     9.81 * np.tan(thetay),
    #     -8 * thetay + omegay,
    #     -10 * thetay + 10 * ay,
    #     vz,
    #     0.91 * az - 9.81
    # ])

    dot_x = np.array([
        vx,
        9.81 * np.tan(thetax),
        -8 * thetax + omegax,
        -10 * thetax + 10 * (3.16227766e+00*((x+(goal[0]-x)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z)))-x) - 4.51663196e+00*vx - 4.25011267e+00*thetax - 1.36015533e+00*omegax + (-8.81028908e-18)*((y+(goal[1]-y)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z))) -y) - 4.23825517e-17*vy - 6.54099623e-17*thetay - (-1.59750882e-17)*omegay + 5.67884552e-17*((z+(goal[2]-z)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z)))-z) - 1.48006779e-16*vz),
        vy,
        9.81 * np.tan(thetay),
        -8 * thetay + omegay,
        -10 * thetay + 10 * ((-2.05880071e-16)*((x+(goal[0]-x)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z)))-x) - (-2.64242752e-16)*vx - (-2.62071672e-1)*thetax - (-1.59750882e-17)*omegax + 1.00000000e+00*((y+(goal[1]-y)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z))) -y) - 1.83224920e+00*vy - 1.46496768e+00*thetay - 1.13709874e+00*omegay + (-1.93212270e-16)*((z+(goal[2]-z)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z)))-z) - (-2.67619048e-16)*vz),
        vz,
        0.91 * (5.56619925e-17*((x+(goal[0]-x)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z)))-x) - 7.37457514e-17*vx - 1.31287250e-16*thetax - 1.34686169e-17*omegax + (-1.02273993e-16)*((y+(goal[1]-y)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z))) -y) - (-1.42693775e-16)*vy - (-1.14242552e-16)*thetay - (-2.43533334e-17)*omegay + 1.00000000e+00*((z+(goal[2]-z)/np.sqrt((goal[0]-x)*(goal[0]-x)+(goal[1]-y)*(goal[1]-y)+(goal[2]-z)*(goal[2]-z)))-z) - 1.78823997e+00*vz + 9.81/0.91) - 9.81
    ])

    # dot_x = np.array([
    #     vx,
    #     9.81 * np.tan(thetax),
    #     -8 * thetax + omegax,
    #     -10 * thetax + 10 * (3.16227766e+00*(goal[0]-x) - 4.51663196e+00*vx - 4.25011267e+00*thetax - 1.36015533e+00*omegax + (-8.81028908e-18)*(goal[1]-y) - 4.23825517e-17*vy - 6.54099623e-17*thetay - (-1.59750882e-17)*omegay + 5.67884552e-17*(goal[2]-z) - 1.48006779e-16*vz),
    #     vy,
    #     9.81 * np.tan(thetay),
    #     -8 * thetay + omegay,
    #     -10 * thetay + 10 * ((-2.05880071e-16)*(goal[0]-x) - (-2.64242752e-16)*vx - (-2.62071672e-1)*thetax - (-1.59750882e-17)*omegax + 1.00000000e+00*(goal[1]-y) - 1.83224920e+00*vy - 1.46496768e+00*thetay - 1.13709874e+00*omegay + (-1.93212270e-16)*(goal[2]-z) - (-2.67619048e-16)*vz),
    #     vz,
    #     0.91 * (5.56619925e-17*(goal[0]-x) - 7.37457514e-17*vx - 1.31287250e-16*thetax - 1.34686169e-17*omegax + (-1.02273993e-16)*(goal[1]-y) - (-1.42693775e-16)*vy - (-1.14242552e-16)*thetay - (-2.43533334e-17)*omegay + 1.00000000e+00*(goal[2]-z) - 1.78823997e+00*vz + 9.81/0.91) - 9.81
    # ])
    return dot_x

def simulate(x, dt):
    # curr_position = np.array(x)[[0,2,4]]
    # error = goal - curr_position
    # distance = np.sqrt((error**2).sum())
    # if distance > 1:
    #     goal = curr_position + error / distance
    return odeint(f, x, [0, dt])[-1]

if __name__ == "__main__":
    for i in range(10):
        trace = [random.uniform(-1,1),0,0,0,random.uniform(-1,1),0,0,0,random.uniform(-1,1),0]
        res = [trace]
        t = 0
        time_list = [t]
        for i in range(2000):
            trace = simulate(trace, 0.01)
            print(trace)
            res.append(trace)
            t += 0.001
            time_list.append(t)

        res = np.array(res)
        plt.figure(0)
        plt.plot(time_list, res[:,0], 'b')
        plt.ylabel('x')

        plt.figure(1)
        plt.plot(time_list, res[:,4], 'b')
        plt.ylabel('y')

        plt.figure(2)
        plt.plot(time_list, res[:,8], 'b')
        plt.ylabel('z')

    plt.show()