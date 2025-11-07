import numpy as np
from compute import residual, lhs
from preprocessing import get_derivatives_and_rhs

def cg_main(image_path1, image_path2, reg, tol=1.e-8, maxit=2000):
    Ix, Iy, rhsu, rhsv = get_derivatives_and_rhs(image_path1, image_path2)
    # Initialize the solution
    u0 = np.zeros_like(Ix)
    v0 = np.zeros_like(Iy)
    # Call the CG solver
    u, v, res, max_iter = OF_cg(u0, v0, Ix, Iy, reg, rhsu, rhsv, tol, maxit)
    return u, v , res, max_iter


def OF_cg(u0,v0,Ix,Iy,reg,rhsu,rhsv,tol=1.e-8,maxit=2000):
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
    # Initialize variables
    u = u0.copy()
    v = v0.copy()
    # Initial residual
    rhu, rhv = residual(u, v, Ix, Iy, reg, rhsu, rhsv)
    p_u = rhu.copy()
    p_v = rhv.copy()
    rs_0 = np.sum(rhu**2) + np.sum(rhv**2)
    rsold = rs_0
    for it in range(maxit):
        Ap_u, Ap_v = lhs(p_u, p_v, Ix, Iy, reg)
        #To avoid division by zero, dont know if this messes up anything
        alpha = rsold / (np.sum(p_u * Ap_u) + np.sum(p_v * Ap_v) + 1e-10)
        u = u + alpha * p_u
        v = v + alpha * p_v
        rhu = rhu - alpha * Ap_u
        rhv = rhv - alpha * Ap_v
        rsnew = np.sum(rhu**2) + np.sum(rhv**2)
        # Check for convergence
        if np.sqrt(rsnew/rsold) < tol:
            break
        p_u = rhu + (rsnew / rsold) * p_u
        p_v = rhv + (rsnew / rsold) * p_v
        rsold = rsnew
        #Note norm of residual is squared
    print(it)
    return u, v, rsnew, it+1
