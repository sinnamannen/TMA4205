import numpy as np
import time

from preprocessing_2 import get_derivatives_and_rhs


def cg_main(image_path1, image_path2, reg, tol=1.e-8, maxit=2000, from_file=False, sigma=0):
    Ix, Iy, rhsu, rhsv = get_derivatives_and_rhs(image_path1, image_path2, from_file=from_file, sigma=sigma)
    # Initialize the solution
    u0 = np.zeros_like(Ix)
    v0 = np.zeros_like(Iy)
    # Call the CG solver
    u, v, res, max_iter, elapsed_time = OF_cg(u0, v0, Ix, Iy, reg, rhsu, rhsv, tol, maxit)
    return u, v , res, max_iter, elapsed_time

def apply_lhs(u, v, Ix, Iy, reg):
    def laplace(u):
        u_pad = np.pad(u, pad_width=1, mode='constant', constant_values=0)
        laplace_u = -(
            4 * u_pad[1:-1, 1:-1]
            - u_pad[1:-1, :-2]
            - u_pad[1:-1, 2:]
            - u_pad[:-2, 1:-1]
            - u_pad[2:, 1:-1]
        )
        return laplace_u
    
    Ixx = Ix**2
    Ixy = Ix * Iy
    Iyy = Iy**2

    Ah_u = laplace(u)
    Ah_v = laplace(v)
    
    lhsu = Ixx*u + Ixy*v - reg*Ah_u
    lhsv = Iyy*v + Ixy*u - reg*Ah_v

    return lhsu, lhsv



def residual(u, v, Ix, Iy, reg, rhsu, rhsv):
    lhsu, lhsv = apply_lhs(u, v, Ix, Iy, reg)
    rhu = rhsu - lhsu
    rhv = rhsv - lhsv
    return rhu, rhv

def OF_cg(u0, v0, Ix, Iy, reg, rhsu, rhsv, tol=1.e-8, maxit=2000):
    """
    The CG method for the optimal flow problem

    input:
        u0 - initial guess for u
        v0 - initial guess for v
        Ix - x-derivative of the first frame
        Iy - y-derivative of the first frame
        reg - regularisation parameter lambda
        rhsu - right-hand side in the equation for u
        rhsv - right-hand side in the equation for v
        tol - relative residual tolerance
        maxit - maximum number of iterations

    output:
        u - numerical solution for u
        v - numerical solution for v
    """
    print("Using Per og Henning")

    def inner_product(pu, pv, qu, qv):
        """ Euclidian inner product for vectors p, q """
        return np.sum(pu * qu) + np.sum(pv * qv)
    
    def norm2(u, v):
        """ Squared 2-norm """
        return inner_product(u, v, u, v)

    start_time = time.time()

    rhu, rhv = residual(u0, v0, Ix, Iy, reg, rhsu, rhsv)
    pu, pv = rhu.copy(), rhv.copy()
    u, v = u0.copy(), v0.copy()

    r0_norm = norm2(rhu, rhv)

    r_list = [1.0]
    
    _iter = 0
    for _iter in range(maxit):
        Apu, Apv = apply_lhs(pu, pv, Ix, Iy, reg)
        alpha = norm2(rhu, rhv) / inner_product(Apu, Apv, pu, pv)

        u += alpha * pu
        v += alpha * pv

        rhu_next = rhu - alpha * Apu
        rhv_next = rhv - alpha * Apv

        beta = norm2(rhu_next, rhv_next) / norm2(rhu, rhv)
        rhu = rhu_next.copy()
        rhv = rhv_next.copy()

        pu = rhu + beta * pu
        pv = rhv + beta * pv
        
        r_norm =  np.sqrt(norm2(rhu, rhv)/r0_norm)
        r_list.append(r_norm)
        if r_norm < tol:
            break

    elapsed_time = time.time() - start_time

    return u, v, r_list,  _iter+1, elapsed_time