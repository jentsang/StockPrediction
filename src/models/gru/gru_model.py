"""
GRU model for stock price prediction.

Architecture diagram: gru_architecture.png (same directory)

Forward pass:
  Input  (batch, seq_len=60, features=18)
    -> GRU Layer 1  [reset gate (r) + update gate (z)]
    -> Dropout (p=0.2)
    -> GRU Layer 2  [reset gate (r) + update gate (z)]
    -> Last time step [:, -1, :]
    -> Dropout (p=0.2)
    -> Linear (128 -> 1)
  Output (batch, 1)  — predicted next close price (normalized)

Key properties:
  - 2 gates: reset (r), update (z)   — no separate cell state
  - ~25% fewer parameters than an equivalent LSTM
  - Faster training, lower memory usage
  - Generalises better on smaller datasets (< 5,000 samples)
  - Accuracy within 1-2% of LSTM on stock benchmarks
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        # GRU only needs hidden state h0 — no cell state c0 unlike LSTM
        h0 = torch.zeros(
            self.num_layers * self.num_directions, x.size(0), self.hidden_size
        ).to(x.device)

        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

    @classmethod
    def from_config(cls, config: dict) -> "GRUModel":
        m = config["model"]
        return cls(
            input_size=m["input_size"],
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            output_size=m["output_size"],
            dropout=m["dropout"],
            bidirectional=m.get("bidirectional", False),
        )
