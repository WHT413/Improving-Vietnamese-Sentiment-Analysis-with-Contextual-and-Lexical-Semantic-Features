from types import SimpleNamespace

import torch


class FakeEncoder(torch.nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch=None, input_ids=None, attention_mask=None):
        ids = batch.input_ids if batch is not None else input_ids
        if ids is None:
            raise ValueError("input_ids or batch is required")
        batch_size, length = ids.shape
        return (
            ids.to(torch.float32).unsqueeze(-1).expand(batch_size, length, self.hidden_size)
            * self.scale
        )
