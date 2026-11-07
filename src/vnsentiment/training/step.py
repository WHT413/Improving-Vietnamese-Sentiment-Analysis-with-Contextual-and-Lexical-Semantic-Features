"""Forward pass plus scaled backward; optimizer stays outside."""

from torch import nn

from vnsentiment.contracts.train_batch import TrainBatch


def train_step(
    model: nn.Module,
    batch: TrainBatch,
    loss_fn: nn.Module | None = None,
    scale: float = 1.0,
) -> float:
    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    logits = model(batch.texts)
    loss = criterion(logits, batch.labels.to(logits.device))
    (loss * scale).backward()
    return float(loss.detach())
