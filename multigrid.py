import numpy as np
from conjugate_gradient import OF_cg
from preprocessing import get_derivatives_and_rhs
from compute import residual

def multigrid_main(image_path1, image_path2, reg, s1, s2, max_level):
    Ix, Iy, rhsu, rhsv = get_derivatives_and_rhs(image_path1, image_path2)
    # Initialize the solution
    u0 = np.zeros_like(Ix)
    v0 = np.zeros_like(Iy)
    # Call the V-cycle
    u, v = V_cycle(u0, v0, Ix, Iy, reg, rhsu, rhsv, s1, s2, 0, max_level)
    return u, v




def V_cycle(u0, v0, Ix, Iy, reg, rhsu, rhsv, s1, s2, level, max_level):
    '''
    V-cycle for the optical flow problem.
    input:
    u0 - initial guess for u
    v0 - initial guess for v
    Ix - x-derivative of the first frame
    Iy - y-derivative of the first frame
    reg - regularisation parameter (lmbda)
    rhsu - right-hand side in the equation for u
    rhsv - right-hand side in the equation for v
    s1 - number of pre-smoothings
    s2 - number of post-smoothings
    level - current level
    max_level - total number of levels
    output:
    u - numerical solution for u
    v - numerical solution for v
    '''
    
    u,v = smoothing(u0, v0, Ix, Iy, reg, rhsu, rhsv, level,s1)
    rhu,rhv = residual(u, v, Ix, Iy, reg, rhsu, rhsv)
    r2hu,r2hv,Ix2h,Iy2h = restriction(rhu, rhv, Ix, Iy)
    
    if level == max_level - 1:
        e2hu,e2hv = OF_cg(np.zeros_like(r2hu), np.zeros_like(r2hv),
            Ix2h, Iy2h, reg, rhsu, rhsv, 1e-8, 1000, level+1)
    else:
        e2hu,e2hv = V_cycle(np.zeros_like(r2hu), np.zeros_like(r2hv),
            Ix2h, Iy2h, reg, r2hu, r2hv, s1, s2, level+1, max_level)
        
    ehu,ehv = prolongation(e2hu, e2hv)
    u = u + ehu
    v = v + ehv
    u,v = smoothing(u, v, Ix, Iy, reg, rhsu, rhsv, level, s2)
    return u, v

def smoothing(u0, v0, Ix, Iy, reg, rhsu, rhsv, level, s):
    '''
    Smoothing for the optical flow problem.
    Using Red-Black Gauss-Seidel smoothing.
    input:
    u0 - initial guess for u
    v0 - initial guess for v
    Ix - x-derivative of the first frame
    Iy - y-derivative of the first frame
    reg - regularisation parameter (lmbda)
    rhsu - right-hand side in the equation for u
    rhsv - right-hand side in the equation for v
    level - current level
    s - number of smoothing iterations
    output:
    u - numerical solution for u
    v - numerical solution for v
    '''
    #Padding u0 and v0 with zeros for boundary conditions
    n,m = u0.shape
    u = np.zeros((n+2,m+2))
    u[1:-1,1:-1] = u0
    v = np.zeros((n+2,m+2))
    v[1:-1,1:-1] = v0
    for _ in range(s):
        u, v = RBGS_step(u, v, Ix, Iy, reg, rhsu, rhsv)
    return u, v

def RBGS_step(u, v, Ix, Iy, reg, rhsu, rhsv):
    '''
    One Red-Black Gauss-Seidel iteration for the optical flow problem.
    input:
    u - current solution for u
    v - current solution for v
    Ix - x-derivative of the first frame
    Iy - y-derivative of the first frame
    reg - regularisation parameter (lmbda)
    rhsu - right-hand side in the equation for u
    rhsv - right-hand side in the equation for v
    n - number of rows
    m - number of columns
    output:
    u - updated solution for u
    v - updated solution for v
    '''
    n,m = u.shape
    #Red update
    #Possible to do this without double for-loop?????????
    for i in range(1,n-1):
            for j in range(1,m-1):
                if (i+j) % 2 == 0:
                    Ix_ij = Ix[i-1,j-1]
                    Iy_ij = Iy[i-1,j-1]
                    denom_u = Ix_ij**2 + reg*4
                    denom_v = Iy_ij**2 + reg*4

                    u[i,j] = (rhsu[i,j] + reg * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]) - Ix_ij*Iy_ij*v[i,j]) / denom_u
                    v[i,j] = (rhsv[i,j] + reg * (v[i-1,j] + v[i+1,j] + v[i,j-1] + v[i,j+1]) - Ix_ij*Iy_ij*u[i,j]) / denom_v
    #Black update
    for i in range(1,n-1):
            for j in range(1,m-1):
                if (i+j) % 2 == 1:
                    Ix_ij = Ix[i-1,j-1]
                    Iy_ij = Iy[i-1,j-1]
                    denom_u = Ix_ij**2 + reg*4
                    denom_v = Iy_ij**2 + reg*4

                    u[i,j] = (rhsu[i,j] + reg * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]) - Ix_ij*Iy_ij*v[i,j]) / denom_u
                    v[i,j] = (rhsv[i,j] + reg * (v[i-1,j] + v[i+1,j] + v[i,j-1] + v[i,j+1]) - Ix_ij*Iy_ij*u[i,j]) / denom_v
    return u, v



def restriction(rhu, rhv, Ix, Iy):
    '''
    Restriction operator for the optical flow problem.
    input:
    rhu - residual for u
    rhv - residual for v
    Ix - x-derivative of the first frame
    Iy - y-derivative of the first frame
    output:
    r2hu - restricted residual for u
    r2hv - restricted residual for v
    Ix2h - restricted x-derivative of the first frame
    Iy2h - restricted y-derivative of the first frame
    '''

    r2hu = 0.25 * (rhu[0::2, 0::2] + rhu[1::2, 0::2] + rhu[0::2, 1::2] + rhu[1::2, 1::2])
    r2hv = 0.25 * (rhv[0::2, 0::2] + rhv[1::2, 0::2] + rhv[0::2, 1::2] + rhv[1::2, 1::2])
    Ix2h = 0.25 * (Ix[0::2, 0::2] + Ix[1::2, 0::2] + Ix[0::2, 1::2] + Ix[1::2, 1::2])
    Iy2h = 0.25 * (Iy[0::2, 0::2] + Iy[1::2, 0::2] + Iy[0::2, 1::2] + Iy[1::2, 1::2])
    
    return r2hu, r2hv, Ix2h, Iy2h

def prolongation(e2hu, e2hv):
    '''
    Prolongation operator for the optical flow problem.
    input:
    e2hu - error on the coarse grid for u
    e2hv - error on the coarse grid for v
    output:
    ehu - prolonged error for u
    ehv - prolonged error for v
    '''
    n,m = e2hu.shape
    ehu = np.zeros((2*n, 2*m))
    ehv = np.zeros((2*n, 2*m))
    
    ehu[0::2, 0::2] = e2hu
    ehu[1::2, 0::2] = e2hu
    ehu[0::2, 1::2] = e2hu
    ehu[1::2, 1::2] = e2hu
    
    return ehu, ehv