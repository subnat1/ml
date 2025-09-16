
import numpy as np
from .simp_core import penalize_k
from ..01_data.gen_random_fields import jacobi_solve

def forward_temp_fea(delta, Q, sink_mask, p=3.0, kmin=0.01, kmax=1.0, iters=2000, tol=1e-5):
    K = penalize_k(delta, kmin=kmin, kmax=kmax, p=p)
    T = jacobi_solve(K.astype(np.float64), Q.astype(np.float64), sink_mask.astype(np.float64), max_iter=iters, tol=tol)
    return T
