"""Union-join the three sources into 12-dimensional entries."""

from typing import Any


def build_entries(
    vswn_data: dict[str, tuple[float, int]],
    vad_data: dict[str, tuple[list[float], list[int]]],
    vnemo_data: dict[str, tuple[list[float], list[int]]],
) -> dict[str, dict[str, list[Any]]]:
    all_keys = sorted(set(vswn_data) | set(vad_data) | set(vnemo_data))
    entries: dict[str, dict[str, list[Any]]] = {}
    for key in all_keys:
        pol_val, pol_mask = vswn_data.get(key, (0.0, 0))
        vad_vals, vad_masks = vad_data.get(key, ([0.0, 0.0, 0.0], [0, 0, 0]))
        emo_vals, emo_masks = vnemo_data.get(key, ([0.0] * 8, [0] * 8))
        values = [float(pol_val)] + [float(x) for x in vad_vals] + [float(x) for x in emo_vals]
        mask = [int(pol_mask)] + [int(m) for m in vad_masks] + [int(m) for m in emo_masks]
        if any(mask):
            entries[key] = {"values": values, "mask": mask}
    return entries