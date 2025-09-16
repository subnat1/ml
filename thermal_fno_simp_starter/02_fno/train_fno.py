
import argparse, numpy as np, torch, torch.nn as nn, torch.optim as optim
from model_fno2d import FNO2d

def load_data(path):
    z = np.load(path)
    K, Q, S, T = z["K"], z["Q"], z["S"], z["T"]
    X = np.stack([K, Q, S], axis=1) # (N, 3, H, W)
    Y = T[:, None, :, :]            # (N, 1, H, W)
    return X, Y

def train(args):
    X, Y = load_data(args.data)
    N = X.shape[0]
    split = int(0.9 * N)
    idx = np.random.permutation(N)
    tr, va = idx[:split], idx[split:]
    Xtr, Ytr = torch.from_numpy(X[tr]).float(), torch.from_numpy(Y[tr]).float()
    Xva, Yva = torch.from_numpy(X[va]).float(), torch.from_numpy(Y[va]).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNO2d(in_channels=3, out_channels=1, width=args.width, modes1=args.modes, modes2=args.modes, layers=args.layers).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def rel_l2(pred, true):
        num = torch.linalg.norm(pred-true)
        den = torch.linalg.norm(true) + 1e-12
        return (num/den).item()

    for ep in range(1, args.epochs+1):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        losses = []
        for i in range(0, Xtr.shape[0], args.bs):
            ii = perm[i:i+args.bs]
            xb = Xtr[ii].to(device); yb = Ytr[ii].to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            predv = model(Xva.to(device))
            val_loss = loss_fn(predv, Yva.to(device)).item()
            val_rel = rel_l2(predv, Yva.to(device))

        print(f"Epoch {ep:03d} | train {np.mean(losses):.4e} | val {val_loss:.4e} | relL2 {val_rel:.4f}")

    if args.save:
        torch.save(model.state_dict(), args.save)
        print("Saved model to", args.save)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--save", type=str, default="02_fno/fno.pt")
    args = ap.parse_args()
    train(args)
