from __future__ import annotations

from typing import Any, Dict


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "PyTorch is required for NAND model operations. Install torch in your runtime (e.g., Colab GPU)."
        ) from exc
    return torch, nn


def create_encoder(config: Dict[str, Any]):
    torch, nn = _require_torch()

    input_dim = int(config.get("input_dim", 818))
    hidden_dim = int(config.get("hidden_dim", 1024))
    output_dim = int(config.get("output_dim", 256))
    dropout = float(config.get("dropout", 0.2))

    class NandEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.SELU(),
            )

        def forward(self, x):
            z = self.net(x)
            return nn.functional.normalize(z, dim=-1)

    return NandEncoder()


def info_nce_loss(anchors, positives, temperature: float = 0.25):
    torch, nn = _require_torch()
    logits = anchors @ positives.T
    logits = logits / max(temperature, 1e-8)
    targets = torch.arange(logits.size(0), device=logits.device)
    return nn.functional.cross_entropy(logits, targets)
