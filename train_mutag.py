from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from lgan.model import LGANGraphClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: LGANGraphClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=-1)
        correct += int((pred == data.y).sum())
        total += data.y.numel()
    return correct / max(total, 1)


def train_one_fold(
    dataset: TUDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    # 폴드마다 시드 고정
    set_seed(args.seed + fold)

    y_all = np.array([int(data.y) for data in dataset])
    y_train = y_all[train_idx]

    # 내부에서 train/val 나누기
    # 테스트는 마지막에만(누수 방지)
    if args.val_ratio > 0.0:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_ratio, random_state=args.seed + fold
        )
        rel_train, rel_val = next(splitter.split(np.zeros(len(train_idx)), y_train))
        inner_train_idx = train_idx[rel_train]
        val_idx = train_idx[rel_val]
    else:
        inner_train_idx = train_idx
        val_idx = None

    train_set = dataset[inner_train_idx.tolist()]
    val_set = dataset[val_idx.tolist()] if val_idx is not None else None
    test_set = dataset[test_idx.tolist()]

    # 보기 편하게 배치=1로
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False) if val_set is not None else None
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = LGANGraphClassifier(
        in_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        num_classes=dataset.num_classes,
        num_layers=args.layers,
        dropout=args.dropout,
        mlp_hidden=args.mlp_hidden if args.mlp_hidden is not None else args.hidden_dim,
        variant=args.variant,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits, data.y)
            loss.backward()
            opt.step()

        if val_loader is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            val_acc = evaluate(model, val_loader, device)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                # 베스트 가중치는 CPU에 저장
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # 마지막에 테스트 한 번만
    test_acc = evaluate(model, test_loader, device)

    if val_loader is not None:
        print(f'    best val acc = {best_val:.4f} @ epoch {best_epoch:03d} -> test acc = {test_acc:.4f}')

    return test_acc
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--val-ratio', type=float, default=0.1,
                   help='Validation ratio inside each fold for model selection (0 disables).')
    p.add_argument('--variant', type=str, default='lgan-res', choices=['lgan', 'lgan-res'],
                   help='Which update to use: LGAN (Eq.5) or LGAN-res (Eq.6-7).')


    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--mlp-hidden', type=int, default=None)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.5)

    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=5e-4)

    args = p.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = os.path.join(os.path.dirname(__file__), 'data')
    dataset = TUDataset(root=root, name='MUTAG')

    # MUTAG는 보통 노드 피처가 있음
    if dataset.num_features == 0:
        raise RuntimeError('MUTAG has no node features in this setup. Consider creating one-hot node labels.')

    y = np.array([int(data.y) for data in dataset])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)

    accs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        acc = train_one_fold(dataset, train_idx, test_idx, fold, args, device)
        accs.append(acc)
        print(f'[fold {fold:02d}] test acc = {acc:.4f}')

    accs = np.array(accs)
    print('---')
    print(f'10-fold mean acc = {accs.mean():.4f} ± {accs.std():.4f}')


if __name__ == '__main__':
    main()
