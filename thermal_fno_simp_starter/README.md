
# FNO-Accelerated SIMP Topology Optimization (2D Heat Conduction)

This is a minimal, *working* starter kit showing how to couple a **Fourier Neural Operator (FNO)**
temperature surrogate with a **SIMP** optimizer for steady 2D conduction on a rectangular grid.

## What’s included
- **01_data/**
  - `gen_random_fields.py`: Generate conductivity fields + source/sink masks and compute ground-truth temperatures with a simple iterative solver (Jacobi). Saves `npz` files.
- **02_fno/**
  - `model_fno2d.py`: Lightweight 2D FNO blocks in PyTorch.
  - `train_fno.py`: Trains the FNO to predict T(x,y) from [K, source, sink] channels.
  - `infer_fno.py`: Loads a trained FNO and predicts T for given inputs.
- **03_opt/**
  - `simp_core.py`: Density filter, SIMP penalization, OC update, compliance.
  - `forward_fea.py`: Forward temperature solve (same core as data gen).
  - `forward_fno.py`: Forward using trained FNO.
  - `run_optimize.py`: Example SIMP optimization toggling FEA vs FNO and mixing sensitivities.

## Quickstart

1. **Create a venv and install deps**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy
```

2. **Generate a tiny dataset (demo)**
```bash
python 01_data/gen_random_fields.py --n 200 --res 64 --out 01_data/demo_A64.npz --mode A
```
Modes:
- `A`: fixed sinks (four corners), single random point source
- `B`: uniform heating, random sink patches
- `C`: single random source, random sinks
- `D`: multiple random sources, random sinks

3. **Train FNO**
```bash
python 02_fno/train_fno.py --data 01_data/demo_A64.npz --epochs 30 --save 02_fno/fno_A64.pt
```

4. **Run optimization**
```bash
python 03_opt/run_optimize.py --res 64 --mode A --fno 02_fno/fno_A64.pt --iters 60 --use_fno --fea_every 10
```
- Add `--fea_only` to force pure FEA forwards (slow but authoritative).
- Try `--res 128` with the **same** FNO model to see super-resolution behavior (demo-scale only).

> NOTE: This is a compact educational starter, geared for CPU. For production speed and stability, replace the Jacobi solver with a proper sparse CG/AMG, increase dataset size, and consider PyTorch Lightning, mixed precision, and batched data loaders.

## Objective & Compliance
We use a simple thermal compliance: `C = sum(T * source_mask)` (i.e., hot sources penalize temperature).
You can swap in alternatives (avg T in hotspots, max T with smooth-approx, etc.).

## License
MIT
