"""
LSTM model for stock price prediction.

Architecture diagram: lstm_architecture.png (same directory)

Forward pass:
  Input  (batch, seq_len=60, features=18)
    -> LSTM Layer 1  [hidden_state (h) + cell_state (c)]
    -> Dropout (p=0.2)
    -> LSTM Layer 2  [hidden_state (h) + cell_state (c)]
    -> Last time step [:, -1, :]
    -> Dropout (p=0.2)
    -> Linear (128 -> 1)
  Output (batch, 1)  — predicted next close price (normalized)

Key properties:
  - 3 gates: input, forget, output
  - Maintains a separate cell state (c) alongside hidden state (h)
  - Better at very long sequences due to the dedicated cell state
  - ~25% more parameters than an equivalent GRU
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
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

        self.lstm = nn.LSTM(
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
        h0 = torch.zeros(
            self.num_layers * self.num_directions, x.size(0), self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * self.num_directions, x.size(0), self.hidden_size
        ).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

    @classmethod
    def from_config(cls, config: dict) -> "LSTMModel":
        m = config["model"]
        return cls(
            input_size=m["input_size"],
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            output_size=m["output_size"],
            dropout=m["dropout"],
            bidirectional=m.get("bidirectional", False),
        )
