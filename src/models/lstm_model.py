"""Deep learning architectures: LSTM/GRU, 1D-CNN-LSTM, Attention-LSTM.

All models accept input [batch, window, features] and output [batch, 1] RUL.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

import torch
from torch import nn

from src.config import Config

logger = logging.getLogger(__name__)


class AttentionPool(nn.Module):
    """Weighted attention over LSTM hidden states to produce a context vector."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [batch, seq, hidden]
        scores = self.attn(hidden_states)  # [batch, seq, 1]
        weights = torch.softmax(scores, dim=1)  # [batch, seq, 1]
        context = (weights * hidden_states).sum(dim=1)  # [batch, hidden]
        return context, weights.squeeze(-1)


class BaseRNN(nn.Module):
    """Stacked LSTM or GRU with dropout and a linear RUL head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        attention: bool = False,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention
        self.num_directions = 2 if bidirectional else 1
        self.rnn: nn.Module
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        rnn_output_size = hidden_size * self.num_directions
        self.attention_pool: Optional[AttentionPool]
        if attention:
            self.attention_pool = AttentionPool(rnn_output_size)
        else:
            self.attention_pool = None

        self.head = nn.Sequential(
            nn.Linear(rnn_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, features]
        rnn_out, h_n = self._rnn_output(x)  # h_n: [num_layers*dirs, batch, hidden]
        if self.attention_pool is not None:
            context, _ = self.attention_pool(rnn_out)
        elif self.bidirectional:
            # Concatenate final forward and backward hidden states of the last layer
            context = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [batch, hidden*2]
        else:
            context = h_n[-1]  # [batch, hidden]
        return cast(torch.Tensor, self.head(context))

    def _rnn_output(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (rnn_out, h_n) handling LSTM vs GRU output tuple differences."""
        outputs = self.rnn(x)
        rnn_out = outputs[0]
        if isinstance(self.rnn, nn.LSTM):
            h_n = outputs[1][0]
        else:
            h_n = outputs[1]
        return rnn_out, h_n


class CNNLSTM(nn.Module):
    """1D-CNN front-end extracting local sensor patterns, feeding a stacked LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        cnn_filters: Optional[list[int]] = None,
        cnn_kernel: int = 3,
        attention: bool = False,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        cnn_filters = cnn_filters or [32, 64]

        layers = []
        in_ch = input_size
        for out_ch in cnn_filters:
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.rnn = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.attention_pool: Optional[AttentionPool]
        if attention:
            self.attention_pool = AttentionPool(hidden_size)
        else:
            self.attention_pool = None

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, features] -> conv expects [batch, channels, seq]
        x = x.transpose(1, 2)
        x = self.cnn(x)  # [batch, cnn_filters[-1], seq]
        x = x.transpose(1, 2)  # [batch, seq, cnn_filters[-1]]
        rnn_out, h_n = self._rnn_output(x)
        if self.attention_pool is not None:
            context, _ = self.attention_pool(rnn_out)
        elif self.rnn.bidirectional:
            context = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            context = h_n[-1]
        return cast(torch.Tensor, self.head(context))

    def _rnn_output(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (rnn_out, h_n) handling LSTM vs GRU output tuple differences."""
        outputs = self.rnn(x)
        rnn_out = outputs[0]
        if isinstance(self.rnn, nn.LSTM):
            h_n = outputs[1][0]
        else:
            h_n = outputs[1]
        return rnn_out, h_n


def build_model(config: Config, input_size: int) -> nn.Module:
    """Factory to instantiate the configured architecture."""
    arch = config.model.architecture.lower()
    common: dict[str, Any] = dict(
        input_size=input_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        bidirectional=config.model.bidirectional,
        attention=config.model.attention,
    )

    model: nn.Module
    if arch == "lstm":
        model = BaseRNN(rnn_type="lstm", **common)
    elif arch == "gru":
        model = BaseRNN(rnn_type="gru", **common)
    elif arch == "cnn_lstm":
        model = CNNLSTM(
            cnn_filters=config.model.cnn_filters,
            cnn_kernel=config.model.cnn_kernel,
            **common,
        )
    elif arch == "attention_lstm":
        common["attention"] = True
        model = BaseRNN(rnn_type="lstm", **common)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    logger.info("Built %s model with input_size=%d", arch, input_size)
    return model
