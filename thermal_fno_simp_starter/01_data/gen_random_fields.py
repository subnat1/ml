
import argparse
import numpy as np

def mean_filter(R, r):
    k = int(r)
    if k < 1:
        return R.copy()
    try:
        from scipy.ndimage import uniform_filter
        return uniform_filter(R, size=2*k+1, mode='nearest')
    except Exception:
        pad = ((k,k),(k,k))
        Rp = np.pad(R, pad, mode='edge')
        out = np.zeros_like(R)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                out[i,j] = Rp[i:i+2*k+1, j:j+2*k+1].mean()
        return out

def jacobi_solve(k_field, q_field, sink_mask, max_iter=2000, tol=1e-5):
    H, W = k_field.shape
    T = np.zeros((H,W), dtype=np.float64)
    k_e = np.zeros_like(k_field); k_w = np.zeros_like(k_field)
    k_n = np.zeros_like(k_field); k_s = np.zeros_like(k_field)
    k_e[:,:-1] = 2*k_field[:,:-1]*k_field[:,1:]/(k_field[:,:-1]+k_field[:,1:] + 1e-12)
    k_w[:,1:]  = 2*k_field[:,1:]*k_field[:,:-1]/(k_field[:,1:]+k_field[:,:-1] + 1e-12)
    k_n[1:,:]  = 2*k_field[1:,:]*k_field[:-1,:]/(k_field[1:,:]+k_field[:-1,:] + 1e-12)
    k_s[:-1,:] = 2*k_field[:-1,:]*k_field[1:,:]/(k_field[:-1,:]+k_field[1:,:] + 1e-12)
    diag = k_e + k_w + k_n + k_s + 1e-12
    fixed = sink_mask > 0.5
    for it in range(max_iter):
        T_old = T.copy()
        Te = np.zeros_like(T); Tw = np.zeros_like(T)
        Tn = np.zeros_like(T); Ts = np.zeros_like(T)
        Te[:,:-1] = T_old[:,1:]
        Tw[:,1:]  = T_old[:,:-1]
        Tn[1:,:]  = T_old[:-1,:]
        Ts[:-1,:] = T_old[1:,:]
        rhs = k_e*Te + k_w*Tw + k_n*Tn + k_s*Ts - q_field
        T = rhs / diag
        T[fixed] = 0.0
        if np.linalg.norm(T - T_old) / (np.linalg.norm(T_old) + 1e-12) < tol:
            break
    return T

def gen_sample(H, W, mode="A", kmin=0.01, kmax=1.0):
    R = np.random.randn(H,W)
    r = np.random.uniform(8,16)
    s = np.random.uniform(4,8)
    Rf = mean_filter(R, r)
    Rn = (Rf - Rf.min())/(Rf.max()-Rf.min()+1e-12)
    delta = 1.0/(1.0 + np.exp(-s*(Rn-0.5)))
    K = kmin + delta*(kmax-kmin)
    src = np.zeros((H,W), dtype=np.float64)
    sink = np.zeros((H,W), dtype=np.float64)
    if mode == "A":
        sink[0,0]=sink[0,-1]=sink[-1,0]=sink[-1,-1]=1.0
        i = np.random.randint(H//4, 3*H//4)
        j = np.random.randint(W//4, 3*W//4)
        src[i,j] = 1.0
        q = src.copy()
    elif mode == "B":
        q = np.ones((H,W), dtype=np.float64)
        for _ in range(np.random.randint(2,6)):
            i = np.random.randint(0,H); j = np.random.randint(0,W)
            sink[i,j]=1.0
    elif mode == "C":
        i = np.random.randint(H//4, 3*H//4)
        j = np.random.randint(W//4, 3*W//4)
        src[i,j] = 1.0
        q = src.copy()
        for _ in range(np.random.randint(2,6)):
            si = np.random.randint(0,H); sj = np.random.randint(0,W)
            sink[si,sj]=1.0
    else:
        q = np.zeros((H,W), dtype=np.float64)
        for _ in range(np.random.randint(2,5)):
            i = np.random.randint(H//6, 5*H//6)
            j = np.random.randint(W//6, 5*W//6)
            q[i,j] = 1.0
        for _ in range(np.random.randint(3,8)):
            si = np.random.randint(0,H); sj = np.random.randint(0,W)
            sink[si,sj]=1.0
    T = jacobi_solve(K, q, sink, max_iter=2000, tol=1e-5)
    return K, q, sink, T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--out", type=str, default="01_data/demo_A64.npz")
    ap.add_argument("--mode", type=str, default="A", choices=["A","B","C","D"])
    args = ap.parse_args()
    H=W=args.res
    Ks = []; Qs = []; Ss = []; Ts = []
    for idx in range(args.n):
        K,q,sink,T = gen_sample(H,W, mode=args.mode)
        Ks.append(K.astype(np.float32))
        Qs.append(q.astype(np.float32))
        Ss.append(sink.astype(np.float32))
        Ts.append(T.astype(np.float32))
    # import numpy as np
    np.savez_compressed(args.out, K=np.stack(Ks), Q=np.stack(Qs), S=np.stack(Ss), T=np.stack(Ts))
    print(f"Saved: {args.out} with {args.n} samples at {args.res}x{args.res}")

if __name__ == "__main__":
    main()
