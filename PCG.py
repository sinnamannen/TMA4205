import time
import numpy as np
import matplotlib.pyplot as plt
from compute import residual, lhs
from image_generation import generate_test_image, mycomputeColor
from multigrid import V_cycle
from preprocessing import calculate_image_derivatives, get_derivatives_and_rhs, get_rhs


def pcg_main(image_path1, image_path2, reg, max_level, s1, s2, tol=1.e-8, maxit=2000):
    Ix, Iy, rhsu, rhsv = get_derivatives_and_rhs(image_path1, image_path2)
    # Initialize the solution
    u0 = np.zeros_like(Ix)
    v0 = np.zeros_like(Iy)
    # Call the CG solver
    u, v, res, max_iter, elapsed_time = OF_pcg(u0, v0, Ix, Iy, reg, rhsu, rhsv, max_level=max_level, s1=s1, s2=s2, tol=1.e-8, maxit=2000)
    return u, v , res, max_iter, elapsed_time


def OF_pcg(u0, v0, Ix, Iy, reg, rhsu, rhsv, max_level=2, s1=3, s2=3, tol=1.e-8, maxit=2000):
    '''
    The CG method for the optimal flow problem
    input:
    u0 - initial guess for u
    v0 - initial guess for v
    Ix - x-derivative of the first frame
    Iy - y-derivative of the first framereg - regularisation parameter lmbda
    rhsu - right-hand side in the equation for u
    rhsv - right-hand side in the equation for v
    tol - relative residual tolerance
    maxit - maximum number of iterations
    output:
    u - numerical solution for u
    v - numerical solution for v
    '''
    start_time = time.time()
    #List to store residual norms
    res_norms = []
    # Initialize variables
    u = u0.copy()
    v = v0.copy()
    # Initial residual
    rhu, rhv = residual(u, v, Ix, Iy, reg, rhsu, rhsv)
    
    z_u, z_v = V_cycle(np.zeros_like(rhu), np.zeros_like(rhv), Ix, Iy, reg, rhu, rhv, s1, s2, level = 0, max_level = max_level)
    
    p_u = z_u.copy()
    p_v = z_v.copy()
    rs_0 = np.sum(rhu**2) + np.sum(rhv**2)
    rsold = np.sum(rhu * z_u) + np.sum(rhv * z_v)
    res_norms.append(np.sqrt(rsold))
    
    for it in range(maxit):
        Ap_u, Ap_v = lhs(p_u, p_v, Ix, Iy, reg)
        alpha = np.sum(rhu * z_u) + np.sum(rhv * z_v) / (np.sum(Ap_u*p_u) + np.sum(Ap_v*p_v))
        
        u = u + alpha * p_u
        v = v + alpha * p_v
        
        rhu = rhu - alpha * Ap_u
        rhv = rhv - alpha * Ap_v
        z_u, z_v = V_cycle(np.zeros_like(rhu), np.zeros_like(rhv), Ix, Iy, reg, rhu, rhv, s1, s2, level = 0, max_level = max_level)
        
        rsnew = np.sum(rhu * z_u) + np.sum(rhv * z_v)
        beta = rsnew / rsold

        res_norms.append(np.sqrt(rsold))
        
        # Check for convergence
        if np.sqrt(rsnew/rs_0) < tol:
            break
        
        p_u = rhu + beta * p_u
        p_v = rhv + beta * p_v
        rsold = rsnew
        #Note norm of residual is squared
    
    elapsed_time = time.time() - start_time
    return u, v, res_norms, it+1, elapsed_time