import numpy as np

def laplacian(u):
    u_padded = np.pad(u, pad_width=1, mode='constant', constant_values=0)
    laplace_u = -4*u + u_padded[:-2,1:-1] + u_padded[2:,1:-1] + u_padded[1:-1,:-2] + u_padded[1:-1,2:]

    return laplace_u

def lhs(u, v, Ix, Iy, reg):
    '''
    Compute the left-hand side for the optical flow problem.
    input:
    u - current solution for u
    v - current solution for v
    Ix - x-derivative of the first frame
    Iy - y-derivative of the first frame
    reg - regularisation parameter (lmbda)
    output:
    LHS - left-hand side of the equation
    '''
    
    lhs_u = Ix * (Ix * u + Iy * v) - reg * laplacian(u)
    lhs_v = Iy * (Ix * u + Iy * v) - reg * laplacian(v)
    return lhs_u, lhs_v

def residual(u, v, Ix, Iy, reg, rhsu, rhsv):
    '''
    Compute the residual for the optical flow problem.
    input:
    u - current solution for u
    v - current solution for v
    Ix - x-derivative of the first frame
    Iy - y-derivative of the first frame
    reg - regularisation parameter (lmbda)
    rhsu - right-hand side in the equation for u
    rhsv - right-hand side in the equation for v
    output:
    rhu - residual for u
    rhv - residual for v
    '''
    lhs_u, lhs_v = lhs(u, v, Ix, Iy, reg)
    rhu = rhsu - lhs_u
    rhv = rhsv - lhs_v
    return rhu, rhv

