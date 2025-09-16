
import numpy as np

def density_filter(delta, r=3):
    k = int(r)
    if k<1: return delta
    pad = ((k,k),(k,k))
    Dp = np.pad(delta, pad, mode='edge')
    out = np.zeros_like(delta)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            out[i,j] = Dp[i:i+2*k+1, j:j+2*k+1].mean()
    return out

def penalize_k(delta, kmin=0.01, kmax=1.0, p=3.0):
    return kmin + (delta**p)*(kmax-kmin)

def compliance(T, Q):
    return float((T*Q).sum())

def oc_update(delta, sens, vol_frac, move=0.1, l1=0.0, l2=1e9, tol=1e-4):
    x = delta.copy()
    # Ensure sens negative where we want to increase delta; add epsilon
    sens = np.clip(sens, -1e6, -1e-6)
    while (l2 - l1)/(l1 + 1e-12) > tol:
        lam = 0.5*(l1+l2)
        x_new = np.clip(delta * np.sqrt(-sens/(lam+1e-12)), delta - move, delta + move)
        x_new = np.clip(x_new, 1e-3, 1.0)
        if x_new.mean() - vol_frac > 0:
            l1 = lam
        else:
            l2 = lam
        x = x_new
    return x

def sens_fea_like(T, Q, delta, p=3.0, kmin=0.01, kmax=1.0):
    gy, gx = np.gradient(T)
    influence = gx**2 + gy**2
    dk_dd = p*(np.maximum(delta,1e-3)**(p-1))*(kmax-kmin)
    sens = -influence * dk_dd
    sabs = np.abs(sens).mean() + 1e-12
    return sens / sabs
