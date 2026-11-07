"""Static paths and schema for the offline composite lexicon build."""

from pathlib import Path

FEATURES: tuple[str, ...] = (
    "polarity",
    "valence",
    "arousal",
    "dominance",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "trust",
    "anticipation",
)

EMOTION_ORDER: tuple[str, ...] = FEATURES[4:]
VERSION = "0.2.0"

ROOT = Path(__file__).resolve().parents[2]
VSWN_PATH = ROOT / "data" / "lexicons" / "vietsentiwordnet" / "VietSentiWordnet_Ver1.3.5.txt"
VAD_PATH = ROOT / "data" / "lexicons" / "nrc-vad" / "Vietnamese-NRC-VAD-Lexicon.txt"
VNEMO_PATH = ROOT / "data" / "lexicons" / "vnemolex" / "VnEmoLex.xlsx"
TRAIN_PATH = ROOT / "data" / "processed" / "train.jsonl"
DEV_PATH = ROOT / "data" / "processed" / "dev.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "lexicons" / "composite_12d" / f"v{VERSION}"
CURATED_PATH = Path(__file__).with_name("curated_vswn.json")