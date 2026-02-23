# LGAN (Line Graph Aggregation Network) minimal PyG reproduction on MUTAG

A minimal, readable implementation of **LGAN** (Eq. 5) and **LGAN-res** (Eq. 6–7) for **graph classification** on **MUTAG** using **PyTorch Geometric**.

This repo is intentionally small/easy-to-read, and follows the paper structure:
- **COMBpair (Eq. 4)**: symmetric node-pair embedding from endpoints (we use `h_u + h_v` → MLP)
- **AGGRt / AGGRn (Eq. 5)**: target–neighbor vs neighbor–neighbor aggregation
- **LGAN-res (Eq. 6–7)**: residual update variant
- **skip-cat + READOUT (Sec. 4.3)**: concatenate all layer outputs, then sum over nodes for graph readout

## Install

You need **PyTorch + PyG**. Installation differs by CUDA/CPU; follow the official PyG install page for your setup.

Extra dependency used here for stratified CV + stratified train/val split:
```bash
pip install scikit-learn
```

## Run

Default (LGAN-res, 10-fold stratified CV, with inner train/val split for model selection):
```bash
python train_mutag.py --epochs 200 --hidden-dim 64 --layers 3 --dropout 0.5
```

Run **LGAN only** (no residual path):
```bash
python train_mutag.py --variant lgan --epochs 200
```

Disable inner validation split (not recommended for fair reporting; keeps only final-epoch model):
```bash
python train_mutag.py --val-ratio 0
```

## Evaluation protocol (no test leakage)

- **Outer loop**: 10-fold stratified CV (same as before).
- **Inner loop**: for each fold, split the training fold into **train/val** (stratified) and pick the best epoch **by validation accuracy**.
- **Test** is evaluated **once per fold** using the best-val checkpoint.

This avoids selecting epochs/models based on the test set.

## Notes

- For simplicity/clarity, the dataloader uses `batch_size=1`.
- The implementation avoids explicitly building a line graph object; it computes the same two multisets directly:
  - incident edges (t,p)
  - edges among neighbors (p,q)
- **Isolated nodes / nodes with no incident edges**: the fused message `z_t` is set to the **zero vector** (as suggested in the paper).
