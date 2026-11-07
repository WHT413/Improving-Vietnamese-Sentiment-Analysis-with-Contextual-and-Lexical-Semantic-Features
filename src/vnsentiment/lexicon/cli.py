"""Interactive lexicon report used by the root CLI."""

from pathlib import Path

import torch

from vnsentiment.phobert_branch.batch import build_batch
from vnsentiment.contracts.backbone_batch import BackboneBatch
from vnsentiment.contracts.lexical_batch import LexicalBatch
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch

from .profile import mean_profile
from .store import LexiconStore


class LexiconCli:
    def __init__(self, store: LexiconStore, branch: LexiconBranch, tokenizer):
        self.store = store
        self.branch = branch
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls, path: str | Path | None = None, tokenizer=None
    ) -> "LexiconCli":
        if tokenizer is None:
            from vnsentiment.phobert_branch.tokenizer import load_phobert_tokenizer

            tokenizer = load_phobert_tokenizer()
        store = LexiconStore.load(path)
        branch = LexiconBranch(store)
        branch.eval()
        return cls(store, branch, tokenizer)

    def print_report(self, text: str) -> None:
        batch = build_batch(self.tokenizer, [text])
        with torch.no_grad():
            features = self.branch.features([text], batch)
            projected = self.branch([text], batch)
        print(
            f"  Lexicon    : {len(self.store)} entries | "
            f"{len(self.store.features)} dims | "
            f"projection={projected.shape[-1]} (chưa train)"
        )
        print("  per-token  : token | word | pol' | vad | emo | mask | s")
        self._print_rows(text, batch, features)
        self._print_profile(features, batch)
        print(
            f"  tensors    : L={_shape(features.values)} | "
            f"s={_shape(features.presence)} | L~={_shape(projected)}"
        )
        print(
            "  note       : profile từ vựng (polarity' 0.5 = neutral), "
            "không phải dự đoán nhãn; projection chưa train."
        )

    def _print_rows(
        self, text: str, batch: BackboneBatch, features: LexicalBatch
    ) -> None:
        words = text.split()
        for position, word_index in enumerate(batch.word_ids[0].tolist()):
            if word_index < 0:
                continue
            word = words[word_index]
            token = self.tokenizer.convert_ids_to_tokens(
                [int(batch.input_ids[0, position])]
            )[0]
            hit = self.store.lookup(word)
            if hit is None:
                print(f"    {token} | {word} | - | -/-/- | 0/0 | mask=0/12 | s=0 | OOV")
                continue
            values, mask = hit
            print(
                "    "
                f"{token} | {word} | {_polarity(values, mask)} | "
                f"{_vad(values, mask)} | "
                f"emo={int(sum(values[4:]))}/{int(sum(mask[4:]))} | "
                f"mask={int(sum(mask))}/12 | "
                f"s={int(features.presence[0, position].item())}"
            )

    def _print_profile(
        self, features: LexicalBatch, batch: BackboneBatch
    ) -> None:
        means, counts = mean_profile(
            features, batch.content_mask, batch.word_ids
        )
        names = ("polarity'",) + self.store.features[1:]
        pvad = " ".join(
            _profile_item(names[i], means[0, i], counts[0, i])
            for i in range(4)
        )
        emotions = " ".join(
            _profile_item(names[i], means[0, i], counts[0, i])
            for i in range(4, 12)
        )
        print(f"  profile P/VAD (theo từ): {pvad}")
        print(f"  profile emo (theo từ): {emotions}")


def _polarity(values, mask) -> str:
    return f"{values[0]:.4f}" if mask[0] else "-"


def _vad(values, mask) -> str:
    return "/".join(
        f"{value:.3f}" if present else "-"
        for value, present in zip(values[1:4], mask[1:4])
    )


def _profile_item(name: str, mean: torch.Tensor, count: torch.Tensor) -> str:
    if int(count.item()) == 0:
        return f"{name}=-"
    return f"{name}={mean.item():.4f}[n={int(count.item())}]"


def _shape(tensor: torch.Tensor) -> str:
    return "x".join(str(size) for size in tensor.shape)