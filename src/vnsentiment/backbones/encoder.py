from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from vnsentiment.contracts.backbone_batch import BackboneBatch

from .tokenizer import DEFAULT_TOKENIZER_DIR


class PhoBertEncoder(nn.Module):
    def __init__(
        self,
        model=None,
        model_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        if model is None:
            path = Path(model_dir) if model_dir is not None else DEFAULT_TOKENIZER_DIR
            if not path.is_dir():
                raise FileNotFoundError(f"PhoBERT model directory not found: {path}")
            model = AutoModel.from_pretrained(str(path), local_files_only=True)
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.hidden_size = int(self.model.config.hidden_size)

    def forward(self, batch: BackboneBatch) -> torch.Tensor:
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = self.device
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, torch.Tensor):
            return outputs
        return outputs.last_hidden_state
