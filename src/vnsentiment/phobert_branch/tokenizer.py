"""Load the local slow PhoBERT tokenizer."""

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_TOKENIZER_DIR = (
    Path(__file__).resolve().parents[3] / "artifacts" / "hf-cache" / "phobert-base"
)


def load_phobert_tokenizer(
    model_dir: str | Path | None = None, *, local_files_only: bool = True
) -> PreTrainedTokenizerBase:
    path = Path(model_dir) if model_dir is not None else DEFAULT_TOKENIZER_DIR
    if not path.is_dir():
        raise FileNotFoundError(f"PhoBERT tokenizer directory not found: {path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), use_fast=False, local_files_only=local_files_only
    )
    if getattr(tokenizer, "is_fast", False):
        raise ValueError("PhoBERT tokenizer must be slow (use_fast=False)")
    return tokenizer
