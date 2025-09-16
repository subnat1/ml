import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "02_fno")))
from model_fno2d import FNO2d
import numpy as np, torch
# from model_fno2d import FNO2d
from simp_core import penalize_k

class FNOInfer:
    def __init__(self, model_path, width=32, modes=16, layers=4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FNO2d(in_channels=3, out_channels=1, width=width, modes1=modes, modes2=modes, layers=layers).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def forward(self, delta, Q, sink_mask, p=3.0, kmin=0.01, kmax=1.0):
        K = penalize_k(delta, kmin=kmin, kmax=kmax, p=p)
        x = np.stack([K, Q, sink_mask], axis=0)[None, ...]  # (1,3,H,W)
        xt = torch.from_numpy(x).float().to(self.device)
        y = self.model(xt).cpu().numpy()[0,0]
        return y
