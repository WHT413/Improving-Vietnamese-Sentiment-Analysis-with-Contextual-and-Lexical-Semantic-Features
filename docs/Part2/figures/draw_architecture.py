#!/usr/bin/env python3
"""Render the Report 2 architecture as PNG and vector SVG.

Usage: python draw_architecture.py [--output-dir PATH] [--dpi 300]
Requires matplotlib. The diagram preserves the proposed residual fusion;
it does not imply measured gate values or guaranteed OOV suppression.
Index i denotes a shared token position after word/subword alignment.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


NAVY = "#183B59"
ORANGE = "#A65A20"
TEAL = "#16736C"
GRAY = "#586574"
LIGHT_BLUE = "#EFF4F9"
LIGHT_ORANGE = "#FCF3E9"
LIGHT_TEAL = "#EDF7F4"


def render(output_dir: Path, dpi: int = 300) -> list[Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "path",  # Keep Vietnamese glyphs portable.
        "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(7.2, 8.6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 12), ylim=(0, 14.3))
    ax.axis("off")
    text_checks = []

    def text(x, y, label, size=10, color=NAVY, bold=False, **kwargs):
        return ax.text(x, y, label, ha="center", va="center", fontsize=size,
                       color=color, fontweight="bold" if bold else "normal",
                       linespacing=1.35, zorder=4, **kwargs)

    def card(x, y, w, h, fill="white", edge=NAVY, lines=()):
        patch = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0,rounding_size=0.13",
                              facecolor=fill, edgecolor=edge, linewidth=1.1,
                              zorder=2)
        ax.add_patch(patch)
        for dy, label, size, color, bold in lines:
            artist = text(x + w / 2, y + dy, label, size, color, bold)
            text_checks.append((artist, x, y, w, h))
        return patch

    def wire(points, color=NAVY, dashed=False, head=True, width=1.3):
        """Route through explicit orthogonal waypoints, ending at box edges."""
        style = (0, (4, 3)) if dashed else "-"
        if len(points) > 2:
            xs, ys = zip(*points[:-1])
            ax.plot(xs, ys, color=color, linewidth=width, linestyle=style,
                    solid_capstyle="round", zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1],
                                    arrowstyle="-|>" if head else "-",
                                    mutation_scale=11, linewidth=width,
                                    color=color, linestyle=style,
                                    shrinkA=0, shrinkB=0, zorder=1))

    def junction(x, y, color):
        ax.plot(x, y, "o", color=color, markersize=3.5, zorder=3)

    # Input and shared preprocessing.
    text(6, 13.86, "TWO-BRANCH ADAPTIVE FUSION", 14, bold=True)
    text(6, 13.42, "Phân tích cảm xúc tiếng Việt", 10, GRAY)
    card(3.4, 12.38, 5.2, .65, LIGHT_BLUE, lines=[
        (.325, "Văn bản đầu vào", 11, NAVY, True),
    ])
    wire([(6, 12.38), (6, 12.08)])
    card(2.1, 11.08, 7.8, 1.0, lines=[
        (.71, "Tiền xử lý văn bản", 11, NAVY, True),
        (.29, "Chuẩn hóa biến thể • Rút gọn ký tự lặp • Tách từ", 9.6, GRAY, False),
    ])
    wire([(6, 11.08), (6, 10.76)], head=False)
    junction(6, 10.76, NAVY)
    wire([(6, 10.76), (3.1, 10.76), (3.1, 10.18)])
    wire([(6, 10.76), (8.9, 10.76), (8.9, 10.18)], ORANGE, True)

    # Two equal-width branches. BPE/lexical positions need a shared mapping.
    card(.7, 8.55, 4.8, 1.63, LIGHT_BLUE, lines=[
        (1.30, "NHÁNH NGỮ CẢNH", 10, NAVY, True),
        (.90, "PhoBERT-base", 12, NAVY, True),
        (.49, "BPE tokenizer + Transformer", 9.4, GRAY, False),
        (.19, "12 tầng • d = 768", 9.4, GRAY, False),
    ])
    card(6.5, 8.55, 4.8, 1.63, LIGHT_ORANGE, ORANGE, lines=[
        (1.30, "NHÁNH TỪ VỰNG", 10, ORANGE, True),
        (.90, "Lexicon Lookup", 12, NAVY, True),
        (.49, "Lexicon mở rộng • Đặc trưng đa chiều", 9.0, GRAY, False),
        (.19, "Căn chỉnh vị trí từ / subword", 9.3, GRAY, False),
    ])
    wire([(3.1, 8.55), (3.1, 8.15)])
    card(1.2, 7.45, 3.8, .70, LIGHT_BLUE, lines=[
        (.35, r"$h_i\in\mathbb{R}^{d}$", 12, NAVY, False),
    ])
    wire([(8.9, 8.55), (8.9, 8.15)], ORANGE, True)
    card(7.0, 7.45, 3.8, .70, LIGHT_ORANGE, ORANGE, lines=[
        (.35, r"$\ell_i\in\mathbb{R}^{k}$", 12, ORANGE, False),
    ])
    wire([(8.9, 7.45), (8.9, 7.10)], ORANGE, True)
    card(6.5, 6.05, 4.8, 1.05, LIGHT_ORANGE, ORANGE, lines=[
        (.76, "Feature Alignment", 10.5, NAVY, True),
        (.34, r"$\tilde{\ell}_i=W_{\mathrm{proj}}\ell_i+b_{\mathrm{proj}}$", 11, ORANGE, False),
    ])

    # Both features feed the gate AND the residual fusion.
    wire([(3.1, 7.45), (3.1, 5.65)], head=False)
    junction(3.1, 5.65, NAVY)
    wire([(3.1, 5.65), (4.65, 5.65), (4.65, 5.25)])
    wire([(8.9, 6.05), (8.9, 5.65)], ORANGE, True, head=False)
    junction(8.9, 5.65, ORANGE)
    wire([(8.9, 5.65), (7.35, 5.65), (7.35, 5.25)], ORANGE, True)
    card(3.5, 4.05, 5, 1.20, LIGHT_TEAL, TEAL, lines=[
        (.91, "Learnable Gate", 11, TEAL, True),
        (.52, r"$g_i=\sigma\!\left(W_g[h_i;\tilde{\ell}_i]+b_g\right)$", 11, NAVY, False),
        (.17, r"$g_i\in(0,1)^d$", 9.5, GRAY, False),
    ])
    # Bypass wires run outside the gate and terminate at the fusion's top.
    wire([(3.1, 5.65), (2.0, 5.65), (2.0, 3.12), (3.5, 3.12), (3.5, 2.86)])
    wire([(8.9, 5.65), (10.0, 5.65), (10.0, 3.12), (8.5, 3.12), (8.5, 2.86)], ORANGE, True)
    text(1.6, 4.20, r"$h_i$", 11, NAVY)
    text(10.55, 4.20, r"$\tilde{\ell}_i\in\mathbb{R}^d$", 10.5, ORANGE)
    wire([(6, 4.05), (6, 2.86)], TEAL)
    text(6.35, 3.50, r"$g_i$", 10.5, TEAL)
    card(2.7, 1.86, 6.6, 1.0, LIGHT_TEAL, TEAL, lines=[
        (.72, "Residual Feature Fusion", 11, TEAL, True),
        (.30, r"$f_i=h_i+g_i\odot\tilde{\ell}_i$", 13, NAVY, False),
    ])
    wire([(6, 1.86), (6, 1.59)], TEAL)
    card(2.0, .70, 8.0, .89, LIGHT_BLUE, lines=[
        (.64, r"Aggregation ($z\in\mathbb{R}^{d}$)  $\rightarrow$  Linear + Softmax", 10, NAVY, True),
        (.25, "Tích cực (Pos)  /  Trung tính (Neu)  /  Tiêu cực (Neg)", 10, NAVY, False),
    ])
    # Small legend, with no unverified claims about learned behavior.
    wire([(1.05, .30), (1.55, .30)], head=False, width=1.4)
    text(3.42, .30, "Biểu diễn ngữ cảnh", 9, GRAY)
    wire([(6.50, .30), (7.00, .30)], ORANGE, True, head=False, width=1.4)
    text(9.0, .30, "Tín hiệu từ vựng", 9, GRAY)

    # Validate actual rendered text bounds rather than relying on wrap=True.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, x, y, w, h in text_checks:
        bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        if not (bounds.x0 >= x + .08 and bounds.x1 <= x + w - .08
                and bounds.y0 >= y + .02 and bounds.y1 <= y + h - .02):
            raise ValueError(f"Text exceeds its box: {artist.get_text()!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        path = output_dir / f"two_branch_adaptive_fusion.{extension}"
        fig.savefig(path, dpi=dpi, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    for path in render(args.output_dir, args.dpi):
        print(path)
