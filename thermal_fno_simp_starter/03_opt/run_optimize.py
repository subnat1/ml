
import argparse, numpy as np
from .simp_core import density_filter, compliance, oc_update, sens_fea_like
from .forward_fea import forward_temp_fea
from .forward_fno import FNOInfer

def build_problem(H, W, mode="A"):
    vol_frac = 0.4
    delta0 = vol_frac * np.ones((H,W), dtype=np.float32)
    Q = np.zeros((H,W), dtype=np.float32)
    S = np.zeros((H,W), dtype=np.float32)
    if mode == "A":
        S[0,0]=S[0,-1]=S[-1,0]=S[-1,-1]=1.0
        i = H//2; j = W//2
        Q[i,j] = 1.0
    elif mode == "B":
        Q[:,:] = 1.0
        for _ in range(4):
            i = np.random.randint(0,H); j = np.random.randint(0,W)
            S[i,j]=1.0
    elif mode == "C":
        i = H//2; j = W//2
        Q[i,j] = 1.0
        for _ in range(5):
            si = np.random.randint(0,H); sj = np.random.randint(0,W)
            S[si,sj]=1.0
    else:
        for _ in range(3):
            i = np.random.randint(H//4, 3*H//4)
            j = np.random.randint(W//4, 3*W//4)
            Q[i,j] = 1.0
        for _ in range(6):
            si = np.random.randint(0,H); sj = np.random.randint(0,W)
            S[si,sj]=1.0
    return delta0, Q, S, vol_frac

def run(args):
    H=W=args.res
    delta, Q, S, vol_frac = build_problem(H,W, mode=args.mode)
    use_fno = args.use_fno and (args.fno is not None)
    if use_fno:
        fno = FNOInfer(args.fno, width=args.width, modes=args.modes, layers=args.layers)
        forward = lambda d: fno.forward(d, Q, S, p=args.p, kmin=args.kmin, kmax=args.kmax)
    else:
        forward = lambda d: forward_temp_fea(d, Q, S, p=args.p, kmin=args.kmin, kmax=args.kmax, iters=2000, tol=1e-5)

    d = delta.copy()
    hist = []
    for it in range(1, args.iters+1):
        df = density_filter(d, r=args.filter_r)
        T = forward(df)
        C = compliance(T, Q)
        hist.append(C)

        if use_fno and (args.fea_every>0) and (it % args.fea_every == 0):
            T_ref = forward_temp_fea(df, Q, S, p=args.p, kmin=args.kmin, kmax=args.kmax, iters=3000, tol=1e-6)
            sens = sens_fea_like(T_ref, Q, df, p=args.p, kmin=args.kmin, kmax=args.kmax)
        else:
            sens = sens_fea_like(T, Q, df, p=args.p, kmin=args.kmin, kmax=args.kmax)

        d = oc_update(df, sens, vol_frac=vol_frac, move=args.move)

        if it % 10 == 0 or it == 1:
            print(f"Iter {it:03d} | C = {C:.6f} | mean(delta)={d.mean():.3f}")

    T_final = forward_temp_fea(density_filter(d, r=args.filter_r), Q, S, p=args.p, kmin=args.kmin, kmax=args.kmax, iters=4000, tol=1e-6)
    C_final = compliance(T_final, Q)
    print(f"Done. Final compliance (FEA): {C_final:.6f}")
    import numpy as np
    np.savez_compressed(args.out, delta=d, Q=Q, S=S, T=T_final, hist=np.array(hist))
    print(f"Saved results to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--mode", type=str, default="A", choices=["A","B","C","D"])
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--filter_r", type=int, default=3)
    ap.add_argument("--p", type=float, default=3.0)
    ap.add_argument("--kmin", type=float, default=0.01)
    ap.add_argument("--kmax", type=float, default=1.0)
    ap.add_argument("--move", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="03_opt/opt_result.npz")
    ap.add_argument("--use_fno", action="store_true")
    ap.add_argument("--fea_only", action="store_true")
    ap.add_argument("--fea_every", type=int, default=10)
    ap.add_argument("--fno", type=str, default=None)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()
    if args.fea_only:
        args.use_fno = False
    run(args)
