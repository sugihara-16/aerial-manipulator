"""Independent objective-error bound for a convex QP with input boxes.

For every feasible z, convexity gives f(z) >= f(x) + grad(f,x).T(z-x).
Adding an arbitrary equality multiplier leaves this bound unchanged when
A z = b. Minimizing its linear part over the box gives an upper bound on
f(x)-f(optimum). The physical constraints must still be checked separately.
"""
import numpy as np


def box_objective_gap(H,g,A,b,lower,upper,x,equality_multiplier):
    gradient=H@x+g+A.T@equality_multiplier
    gap=np.sum(np.maximum(gradient,0.)*np.maximum(x-lower,0.)+
               np.maximum(-gradient,0.)*np.maximum(upper-x,0.))
    return float(gap+abs(equality_multiplier@(A@x-b)))
