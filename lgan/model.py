from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_undirected, coalesce


def mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """2층 MLP(ReLU+Dropout)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass
class EdgeIndexCache:
    """작은 무방향 그래프용 캐시."""

    edges: torch.Tensor  # [m,2] 중복 없는 무방향 엣지(u<v)
    edge_map: Dict[Tuple[int, int], int]  # (u,v) -> 엣지 id
    neighbors: List[List[int]]  # neighbors[t] = t의 이웃들


def build_cache(edge_index: torch.Tensor, num_nodes: int) -> EdgeIndexCache:
    """무방향 엣지/맵/이웃 리스트 만들기."""
    # 무방향으로 만들고 중복 제거
    ei = to_undirected(edge_index, num_nodes=num_nodes)
    ei, _ = coalesce(ei, None, num_nodes, num_nodes)

    # 무방향 엣지 1개만(u<v)
    mask = ei[0] < ei[1]
    edges = ei[:, mask].t().contiguous()  # [m,2]

    edge_map: Dict[Tuple[int, int], int] = {}
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]

    edges_list = edges.tolist()
    for idx, (u, v) in enumerate(edges_list):
        edge_map[(u, v)] = idx
        neighbors[u].append(v)
        neighbors[v].append(u)

    return EdgeIndexCache(edges=edges, edge_map=edge_map, neighbors=neighbors)


class LGANLayer(nn.Module):
    """단일 그래프용 LGAN 레이어.

AGGRt(타깃-이웃)랑 AGGRn(이웃-이웃) 둘 다 계산함.
use_residual=True면 lgan-res(식 6-7), False면 lgan(식 5).
"""

    def __init__(self, hidden_dim: int, mlp_hidden: int, dropout: float, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual

        # COMBpair(식 4)
        self.edge_mlp = mlp(hidden_dim, mlp_hidden, hidden_dim, dropout)

        # (AGGRt, AGGRn) 합쳐서 MLP
        self.fuse_mlp = mlp(2 * hidden_dim, mlp_hidden, hidden_dim, dropout)

        if self.use_residual:
            # residual 선형변환 W^(l)
            self.res_lin = nn.Linear(hidden_dim, hidden_dim)
            # residual 더한 뒤 MLP(식 7)
            self.post_mlp = mlp(hidden_dim, mlp_hidden, hidden_dim, dropout)
        else:
            self.res_lin = None
            self.post_mlp = None

        self.dropout = dropout

    def forward(self, h: torch.Tensor, cache: EdgeIndexCache) -> torch.Tensor:
        """h: [n,d] -> h_new: [n,d]"""
        n, d = h.size()
        device = h.device

        edges = cache.edges.to(device)

        # COMBpair로 엣지 임베딩 만들기
        u = edges[:, 0]
        v = edges[:, 1]
        h_e = self.edge_mlp(h[u] + h[v])  # [m,d]

        # AGGRt/AGGRn(식 5)
        aggr_t = torch.zeros((n, d), device=device)
        aggr_n = torch.zeros((n, d), device=device)

        edge_map = cache.edge_map  # 파이썬 dict 조회
        neighbors = cache.neighbors

        # 고립 노드는 메시지 0으로
        isolated = torch.zeros((n,), dtype=torch.bool, device=device)

        # 이웃-이웃은 모든 쌍 체크
        # 노드당 비용 O(dv+dv^2)
        for t in range(n):
            nbrs = neighbors[t]
            if len(nbrs) == 0:
                isolated[t] = True
                continue

            # 타깃-이웃 (t,p)
            for p in nbrs:
                a, b = (t, p) if t < p else (p, t)
                eid = edge_map.get((a, b), None)
                if eid is not None:
                    aggr_t[t] += h_e[eid]

            # 이웃-이웃 (p,q)
            ln = len(nbrs)
            if ln >= 2:
                for i in range(ln):
                    pi = nbrs[i]
                    for j in range(i + 1, ln):
                        qj = nbrs[j]
                        a, b = (pi, qj) if pi < qj else (qj, pi)
                        eid = edge_map.get((a, b), None)
                        if eid is not None:
                            aggr_n[t] += h_e[eid]

        z = self.fuse_mlp(torch.cat([aggr_t, aggr_n], dim=-1))  # fusion(식 5/6)
        if isolated.any():
            z = z.clone()
            z[isolated] = 0.0

        if self.use_residual:
            assert self.res_lin is not None and self.post_mlp is not None
            h_new = self.post_mlp(self.res_lin(h) + z)  # 식 7
        else:
            h_new = z  # 식 5
        return h_new
class LGANGraphClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        mlp_hidden: int | None = None,
        variant: str = 'lgan-res',
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = hidden_dim


        variant_norm = variant.strip().lower()
        if variant_norm not in {'lgan', 'lgan-res'}:
            raise ValueError("variant must be one of {'lgan', 'lgan-res'}")
        use_residual = variant_norm == 'lgan-res'

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                LGANLayer(hidden_dim=hidden_dim, mlp_hidden=mlp_hidden, dropout=dropout, use_residual=use_residual)
                for _ in range(num_layers)
            ]
        )

        # 스킵-캣 리드아웃(섹션 4.3)
        self.classifier = mlp(num_layers * hidden_dim, mlp_hidden, num_classes, dropout)

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """그래프 1개 forward.

x: [n, in_dim]
edge_index: [2, m_dir]
리턴: logits [1, num_classes]
"""
        h = F.relu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        cache = build_cache(edge_index=edge_index, num_nodes=h.size(0))

        hs: List[torch.Tensor] = []
        for layer in self.layers:
            h = layer(h, cache)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)

        # Hskip: [n, L*d]
        hskip = torch.cat(hs, dim=-1)

        # READOUT은 노드 합
        hg = hskip.sum(dim=0, keepdim=True)  # [1, L*d]

        out = self.classifier(hg)
        return out
