import numpy as np
from conjugate_gradient import OF_cg
from preprocessing import get_derivatives_and_rhs
from compute import residual
import time

def multigrid_main(img1, img2, reg, s1, s2, max_level, from_file=False, sigma=0):
    Ix, Iy, rhsu, rhsv = get_derivatives_and_rhs(img1, img2, from_file=from_file, sigma=sigma)
    # Initialize the solution
    u0 = np.zeros_like(Ix)
    v0 = np.zeros_like(Iy)
    # Call the V-cycle
    u, v = V_cycle(u0, v0, Ix, Iy, reg, rhsu, rhsv, s1, s2, 0, max_level)
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
    
    reg = reg * 4**(-level)
    #reg = reg * 4**(level)
    denom_u = Ix**2 + reg*4
    denom_v = Iy**2 + reg*4
    Ixy = Ix * Iy
    #Padding u0 and v0 with zeros for boundary conditions
    n,m = u0.shape
    u = np.pad(u0, pad_width=1, mode='constant', constant_values=0)
    v = np.pad(v0, pad_width=1, mode='constant', constant_values=0)
    for _ in range(s):
        u, v = RBGS_step(u, v, denom_u, denom_v, Ixy, reg, rhsu, rhsv)
    return u[1:-1,1:-1], v[1:-1,1:-1]


def RBGS_step(u, v, denom_u, denom_v, Ixy, reg, rhsu, rhsv):
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
    
    def red_update(u, v, denom_u, Ixy, reg, rhsu):
        #red 1
        u[1:-1:2,1:-1:2] = (rhsu[::2, ::2] + reg * (u[0:-2:2,1:-1:2] + u[2::2,1:-1:2] + u[1:-1:2,0:-2:2] + u[1:-1:2,2::2]) - Ixy[::2, ::2]*v[1:-1:2,1:-1:2]) / denom_u[::2, ::2]
        #red 2
        u[2:-1:2,2:-1:2] = (rhsu[1::2, 1::2] + reg * (u[1:-2:2,2:-1:2] + u[3::2,2:-1:2] + u[2:-1:2,1:-2:2] + u[2:-1:2,3::2]) - Ixy[1::2, 1::2]*v[2:-1:2,2:-1:2]) / denom_u[1::2, 1::2]
        return u
    
    def black_update(u, v, denom_u, Ixy, reg, rhsu):
        #black 1
        u[1:-1:2,2:-1:2] = (rhsu[::2, 1::2] + reg * (u[0:-2:2,2:-1:2] + u[2::2,2:-1:2] + u[1:-1:2,1:-2:2] + u[1:-1:2,3::2]) - Ixy[::2, 1::2]*v[1:-1:2,2:-1:2]) / denom_u[::2, 1::2]
        #black 2
        u[2:-1:2,1:-1:2] = (rhsu[1::2, ::2] + reg * (u[1:-2:2,1:-1:2] + u[3::2,1:-1:2] + u[2:-1:2,0:-2:2] + u[2:-1:2,2::2]) - Ixy[1::2, ::2]*v[2:-1:2,1:-1:2]) / denom_u[1::2, ::2]
        return u
    
    ##Forward and backward sweep to maintain symmetry i.e. red-black-black-red
    u = red_update(u, v, denom_u, Ixy, reg, rhsu)
    v = red_update(v, u, denom_v, Ixy, reg, rhsv)
    
    u = black_update(u, v, denom_u, Ixy, reg, rhsu)
    v = black_update(v, u, denom_v, Ixy, reg, rhsv)
    
    u = black_update(u, v, denom_u, Ixy, reg, rhsu)
    v = black_update(v, u, denom_v, Ixy, reg, rhsv)
    
    u = red_update(u, v, denom_u, Ixy, reg, rhsu)
    v = red_update(v, u, denom_v, Ixy, reg, rhsv)
    
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
    m,n = e2hu.shape
    ehu = np.zeros((2*m, 2*n))
    ehv = np.zeros((2*m, 2*n))
    
    ehu[0::2, 0::2] = e2hu
    ehu[1::2, 0::2] = e2hu
    ehu[0::2, 1::2] = e2hu
    ehu[1::2, 1::2] = e2hu
    
    ehv[0::2, 0::2] = e2hv
    ehv[1::2, 0::2] = e2hv
    ehv[0::2, 1::2] = e2hv
    ehv[1::2, 1::2] = e2hv
    
    return ehu, ehv



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
    rhu,rhv = residual(u, v, Ix, Iy, reg * 4**(-level), rhsu, rhsv)
    #rhu,rhv = residual(u, v, Ix, Iy, reg * 4**(level), rhsu, rhsv)
    r2hu,r2hv,Ix2h,Iy2h = restriction(rhu, rhv, Ix, Iy)
    if level == max_level - 1:
        cg_reg = reg * 4**(-max_level)
        #cg_reg = reg * 4**(max_level)
        
        e2hu,e2hv, _, _, _ = OF_cg(np.zeros_like(r2hu), np.zeros_like(r2hv),
            Ix2h, Iy2h, cg_reg, r2hu, r2hv, 1e-8, 1000)
    else:
        e2hu,e2hv = V_cycle(np.zeros_like(r2hu), np.zeros_like(r2hv),
            Ix2h, Iy2h, reg, r2hu, r2hv, s1, s2, level+1, max_level)
    ehu,ehv = prolongation(e2hu, e2hv)
    u = u + ehu
    v = v + ehv
    u,v = smoothing(u, v, Ix, Iy, reg, rhsu, rhsv, level, s2)
    return u, v
  


def multigrid_main_iterative(img1, img2, reg, s1, s2, max_level, from_file=False, sigma=0, tol=1e-8, max_cycles=100):
    """
    Full iterative MG solver performing repeated V-cycles.
    Returns:
        u, v          final solution
        res_list     residual history
        elapsed      total computation time
    """
    # Compute derivatives + RHS
    Ix, Iy, rhsu, rhsv = get_derivatives_and_rhs(
        img1, img2, from_file=from_file, sigma=sigma
    )

    # Initial guess
    u = np.zeros_like(Ix)
    v = np.zeros_like(Iy)

    # Compute initial residual
    rhu, rhv = residual(u, v, Ix, Iy, reg, rhsu, rhsv)
    r0 = np.sqrt(np.sum(rhu**2) + np.sum(rhv**2))

    res_list = [1.0]  # normalized residual

    # Start timing
    start = time.time()

    for cycle in range(max_cycles):

        # Perform a single V-cycle
        u, v = V_cycle(u, v, Ix, Iy, reg, rhsu, rhsv,
                       s1, s2, level=0, max_level=max_level)

        # Compute residual
        rhu, rhv = residual(u, v, Ix, Iy, reg, rhsu, rhsv)
        r = np.sqrt(np.sum(rhu**2) + np.sum(rhv**2))
        res_list.append(r / r0)

        # Check convergence
        if r / r0 < tol:
            break

    elapsed = time.time() - start
    return u, v, res_list, elapsed