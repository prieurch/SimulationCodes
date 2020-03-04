# =============================================================================
# File: heat.py
# Author: C Prieur
# for a joint work with Patricio Guzman
# Date: February 2020
# 
# To discretize a heat equation
# of the form
# TO BE WRITTEN
# z_t - a * z_xx - c * z = u + d
# z(t,0)= z(t,L)=0
# z(0,x)= z0(x)
# where 
# u is the control
# d is the disturbance
# c is the positive value
# z0 is the initial condition
# L is the length

# Some intermediate functions
# Solvers
# Initial condition
# Lyapunov function 
# 
# Outputs
# Some figures
# The solution is in z_tot
# 
# =============================================================================


import numpy as np
import time
import sys
#from scitools.std import *
#import scipy.sparse
#import scipy.sparse.linalg
import numpy.linalg as la


def solver_FE(u0, d, L, a, c, Nx, F, T):
    """
    Forward Euler method. 
    Arguments: 
        u0 = initial condition
        d  = disturbance
        k  = control
        L  = length of the space domain
        a  = diffusion
        c  = reaction term
        Nx = number of space discretization
        F  = numerical value. Should be smaller than 0.5
        T  = legnth of the time domain
    Outputs:
        u_tot = numerical solution
        x     = space discretization
        t     = time discretization
        cpu   = computation time
    """
    t0 = time.process_time()
    x = np.linspace(0, L, Nx)   # mesh points in space
    dx = x[1] - x[0]
    dt = F*dx**2/a
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt)   # mesh points in time
    u   = np.zeros(Nx)
    u_1 = np.zeros(Nx)
    
    
    # to save the solution
    n=1 # number of components; n=1 in all the document
    u_tot=np.zeros((Nt,n,Nx))
    control_tot=np.zeros((Nt,n,Nx))

    # Set initial condition u(x,0) = I(x)
    for i in range(0, Nx):
        u_1[i] = u0(x[i])
    u_tot[0,0,:]=u_1

    for tt in range(1, Nt):
        # Compute u at inner mesh points
        for i in range(1, Nx-1):
            u[i] = u_1[i] + F*(u_1[i-1] - 2*u_1[i] + u_1[i+1])
            u[i] = u[i] + dt*( c*u_1[i]+ d(i*dx,tt*dt)+k1(i*dx,u_1)+k2(i,u_1))
            control_tot[tt-1,0,i]=k1(i*dx,u_1)+k2(i,u_1)
        # Insert boundary conditions
        u[0] = 0
        u[Nx-1] = 0

        # Switch variables before next step
        u_tot[tt,0,:]=u
        u_1, u = u, u_1
        
    control_tot[Nt-1,:,:]=control_tot[Nt-2,:,:] # for the last value control
    t1 = time.process_time()
    return u_tot, control_tot, x, t, t1-t0

def viz(uu_tot,x, t,name,choices=['2D','final','3D']):
    import matplotlib.pyplot as plt
    t0 = time.process_time()
    nfig=-1
    n=1    # number of equation to be discretized
    for choice in choices:
        if choice=='2D': # one curve for each x
            nfig+=1; plt.figure(nfig)
            for i in range(0, np.int(len(t)/10), 2):
                for nn in range(n): 
                    plt.plot(x, uu_tot[i][nn,:], label='t={0:1.2f}'.format(t[i]))
                    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.xlabel('x',fontsize=ft_sz)
                    plt.ylabel(name+'(:,x)',fontsize=ft_sz)
            plt.legend()
            plt.savefig(name+'xt.png')
            nfig+=1; plt.figure(nfig)
            for i in range(2*np.int(len(t)/10),len(t), 3):
                for nn in range(n): 
                    plt.plot(x, uu_tot[i][nn,:], label='t={0:1.2f}'.format(t[i]))
                    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.xlabel('x',fontsize=ft_sz)
                    plt.ylabel(name+'(:,x)',fontsize=ft_sz)
            #plt.legend()
            plt.savefig(name+'xt2.png')
               
        if choice=='final': # only the final state is given
            nfig+=1; 
            plt.figure()
            for nn in range(n):
                #plt.plot(x, u_tot[-1][nn,:])
                plt.plot(x, uu_tot[-1][nn,:])
                plt.xlabel('x')
                plt.ylabel('final '+name+'(x)',fontsize=ft_sz) 
                plt.subplots_adjust(top=0.89, right=0.77)
                plt.savefig(name+'T.png')
                plt.show()
                
        if choice=='initial': # only the initial state is given
            nfig+=1; 
            plt.figure()
            for nn in range(n):
                # plt.plot(x, u0(x))
                plt.plot(x, uu_tot[0][nn,:])
                plt.xlabel('x')
                plt.ylabel('initial '+name+'(x)',fontsize=ft_sz) 
                plt.subplots_adjust(top=0.89, right=0.77)
                plt.savefig(name+'0.png')
                #plt.show()        
        
        if choice=='3D': # one curve for each x
            from mpl_toolkits.mplot3d import Axes3D
            nfig+=1; fig=plt.figure()#plt.figure(figsize=(10,4))
            # figure 3D : u(x,t)
            for nn in range(n):
                ax = fig.add_subplot(111, projection='3d')
                SX, ST = np.meshgrid(x, t)
                ax.plot_surface(SX, ST, uu_tot[:,nn,:], cmap='jet')
                ax.set_xlabel('x',fontsize=ft_sz)
                ax.set_ylabel('t',fontsize=ft_sz)
                ax.zaxis.set_rotate_label(False)  # disable automatic rotation
                ax.set_zlabel(name+'(t,x)',fontsize=ft_sz,rotation=90)
                ax.view_init(elev=15, azim=20) # adjust view so it is easy to see
                plt.show()
                plt.savefig(name+'3D.png')
                
    t1 = time.process_time()
    return t1-t0

def sign(ii,u):
    # sign function
    dx=L/(Nx-1)
    int1=np.sqrt(sum(u*u)/dx)
    if int1== 0:
        out=0
    else:
        out=u[ii]/int1
    return out
    
def d(xx,t):
    # this is the distributed disturbance 
    return D*(np.sin(t)*np.sin(xx*2*np.pi/L))

def phi(xx,n):
    # nth eigenvector
    return np.sqrt(2/L)*np.sin(n*np.pi*xx/L)

def k1(xx,u):
    # this is the first part of the control
    x = np.linspace(0, L, Nx);dx = x[1] - x[0] 
    int1=sum(phi(x,1)*u)/dx*phi(xx,1)
    int2=sum(phi(x,2)*u)/dx*phi(xx,2)
    control1 = (-omega-lambda1)*int1
    control1 = control1 + (-omega-lambda2)* int2
    return control1
    
def k2(ii,u):
    # this is the second part of the control
    return -D*sign(ii,u)
    
def u0(xx):
    # this is the initial condition
    return -xx*(2*L/3-xx)*(L-xx)
    # return phi(x,1)


if __name__ == '__main__':
    L=2*np.pi   # length of the space domain
    T=10       # length of the time
    # we use the data of the paper Automatica 109 (2019) 108551 by Lhachemi et al
    a=0.5       # diffusivity
    c=0.5       # reaction term
    Nx=20       # number of space points
    F=0.4       # numerical coefficient
    n=1         # number of equation to be discretized
    D=1.25         # maximal amplitude of the disturbance
    omega=0.5   # stability margin (should be positive)
    lambda1=0.375#first unstable eigenvalue
    lambda2=0   #second unstable eigenvalue
    
    u_tot,control_tot,x,t,cpu = solver_FE(u0, d, L, a, c, Nx, F, T)
    print('CPU time:', cpu)
    
    ft_sz=15 # to set the fontsizes in all plots
    #fig = viz(u_tot,x, t,'z',['3D','initial','final','2D'])
    #fig = viz(control_tot,x, t,'u',['3D','initial','final','2D'])
    fig = viz(control_tot,x, t,'u',['2D'])
    print('CPU time:', cpu)