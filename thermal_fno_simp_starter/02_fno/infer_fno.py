
import argparse, numpy as np, torch
from model_fno2d import FNO2d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--in", dest="inp", type=str, required=True, help="npz with K,Q,S arrays")
    ap.add_argument("--out", type=str, default="T_pred.npy")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    z = np.load(args.inp)
    K, Q, S = z["K"], z["Q"], z["S"]
    X = np.stack([K, Q, S], axis=1) # (N,3,H,W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNO2d(in_channels=3, out_channels=1, width=args.width, modes1=args.modes, modes2=args.modes, layers=args.layers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    with torch.no_grad():
        Xten = torch.from_numpy(X).float().to(device)
        Y = model(Xten).cpu().numpy()  # (N,1,H,W)

    np.save(args.out, Y[:,0])
    print("Saved predictions to", args.out)

if __name__ == "__main__":
    main()
