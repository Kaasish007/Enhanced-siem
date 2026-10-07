from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


INPUT_DIM = 14


class Autoencoder(nn.Module):
    """
    Architecture used during training (see `app.py` in the original script).
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass(frozen=True)
class ModelPaths:
    autoencoder_state_path: Path


def load_autoencoder(model_path: str | Path) -> Optional[Autoencoder]:
    """
    Load the trained autoencoder weights from disk.
    Returns `None` if the model file is missing.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        return None

    model = Autoencoder(input_dim=INPUT_DIM)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

