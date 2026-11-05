"""Recreate the supplied diagram with only its orange bypass label corrected.

Run: python draw_architecture_corrected.py [--output-dir PATH] [--dpi 300]
The original SVG and report remain unchanged. No model behavior is simulated.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLUE = "#173f63"
ORANGE = "#b2601d"
TEAL = "#167c78"
GRAY = "#667788"
PALE_BLUE = "#edf4fa"
PALE_ORANGE = "#fff5ea"
PALE_TEAL = "#eaf7f5"
CORRECTED_LABEL = r"$\tilde{l}_i$: vector từ vựng đã căn chỉnh"


def render(output_dir: Path, dpi: int = 300) -> list[Path]:
    """Use the reference SVG's 1000 x 1240 layout coordinates unchanged."""
    plt.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "dejavuserif",
        "svg.fonttype": "path",
        "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(10, 12.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1000), ylim=(1240, 0))
    ax.axis("off")
    labels = []

    def text(x, y, label, size=18, color=GRAY, bold=False, align="center"):
        artist = ax.text(
            x, y, label, fontsize=size * .72, color=color,
            fontweight="bold" if bold else "normal", ha=align,
            va="baseline", zorder=4,
        )
        labels.append(artist)
        return artist

    def card(x, y, width, height, fill, edge=BLUE, radius=12):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=2.16, zorder=2,
        ))

    def wire(points, color=BLUE, dashed=False):
        style = (0, (7.2, 5.76)) if dashed else "-"
        if len(points) > 2:
            xs, ys = zip(*points[:-1])
            ax.plot(xs, ys, color=color, linewidth=2.16,
                    linestyle=style, solid_capstyle="butt", zorder=1)
        ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>",
            mutation_scale=14, linewidth=2.16, color=color,
            linestyle=style, shrinkA=0, shrinkB=0, zorder=1,
        ))

    text(500, 45, "TWO-BRANCH ADAPTIVE FUSION", 29, BLUE, True)
    text(500, 78, "Phân tích cảm xúc tiếng Việt", 20)
    card(270, 105, 460, 65, PALE_BLUE)
    text(500, 145, "Văn bản đầu vào", 21, BLUE, True)
    wire([(500, 170), (500, 205)])
    card(145, 205, 710, 92, "white")
    text(500, 245, "Tiền xử lý văn bản", 21, BLUE, True)
    text(500, 275, "Chuẩn hóa biến thể • Rút gọn ký tự lặp • Tách từ")
    wire([(500, 297), (500, 330), (275, 330), (275, 350)])
    wire([(500, 330), (725, 330), (725, 350)], ORANGE, True)

    card(55, 350, 440, 130, PALE_BLUE)
    text(275, 390, "NHÁNH NGỮ CẢNH", 21, BLUE, True)
    text(275, 423, "PhoBERT-base", 21, BLUE, True)
    text(275, 454, "BPE tokenizer + Transformer • d = 768")
    card(505, 350, 440, 130, PALE_ORANGE, ORANGE)
    text(725, 390, "NHÁNH TỪ VỰNG", 21, BLUE, True)
    text(725, 423, "Lexicon Lookup", 21, BLUE, True)
    text(725, 454, "Lexicon mở rộng • Đặc trưng đa chiều")

    wire([(275, 480), (275, 510)])
    wire([(725, 480), (725, 510)], ORANGE, True)
    card(125, 510, 300, 58, PALE_BLUE, radius=10)
    text(275, 548, r"$h_i \in \mathbb{R}^{d}$", 22, BLUE)
    card(575, 510, 300, 58, PALE_ORANGE, ORANGE, 10)
    text(725, 548, r"$l_i \in \mathbb{R}^{k}$", 22, BLUE)
    wire([(725, 568), (725, 595)], ORANGE, True)
    card(545, 595, 360, 92, PALE_ORANGE, ORANGE, 10)
    text(725, 630, "Feature Alignment", 21, BLUE, True)
    text(725, 663, r"$\tilde{l}_i = W_{\mathrm{proj}} l_i + b_{\mathrm{proj}}$", 22, BLUE)

    wire([(275, 568), (275, 735), (385, 735), (385, 760)])
    wire([(725, 687), (725, 735), (615, 735), (615, 760)], ORANGE, True)
    text(294, 713, r"$h_i$ vào gate", 16, GRAY, align="left")
    text(666, 713, r"$\tilde{l}_i$ vào gate", 16, GRAY, align="left")
    card(300, 760, 400, 110, PALE_TEAL, TEAL)
    text(500, 800, "Learnable Gate", 21, BLUE, True)
    text(500, 834, r"$g_i = \sigma(W_g[h_i\,;\,\tilde{l}_i] + b_g)$", 22, BLUE)
    text(500, 858, r"Chỉ tạo hệ số điều tiết $g_i$", 16)

    card(250, 1000, 500, 115, PALE_TEAL, TEAL)
    text(500, 1040, "Residual Feature Fusion", 21, BLUE, True)
    text(500, 1075, r"$f_i = h_i + g_i \odot \tilde{l}_i$", 22, BLUE)
    text(500, 1099, r"Nhận trực tiếp $h_i$, $\tilde{l}_i$ và hệ số $g_i$", 16)
    wire([(500, 870), (500, 1000)], TEAL)
    text(520, 936, r"$g_i$", 16, GRAY, align="left")
    wire([(275, 568), (275, 925), (350, 925), (350, 1000)])
    wire([(725, 687), (725, 925), (650, 925), (650, 1000)], ORANGE, True)
    ax.add_patch(FancyBboxPatch(
        (240, 940), 205, 30, boxstyle="round,pad=0,rounding_size=6",
        facecolor="white", edgecolor="none", zorder=3,
    ))
    text(342, 961, r"$h_i$ truyền thẳng vào phép cộng", 16)
    ax.add_patch(FancyBboxPatch(
        (555, 940), 250, 30, boxstyle="round,pad=0,rounding_size=6",
        facecolor="white", edgecolor="none", zorder=3,
    ))
    # Only the content of this bypass annotation differs from the reference.
    corrected_artist = text(680, 961, CORRECTED_LABEL, 16)

    wire([(500, 1115), (500, 1150)], TEAL)
    card(145, 1150, 710, 58, PALE_BLUE, radius=10)
    text(500, 1185, "Aggregation → Linear + Softmax → Pos / Neu / Neg", 21, BLUE, True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in labels:
        bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        if bounds.x0 < 0 or bounds.x1 > 1000 or bounds.y0 < 0 or bounds.y1 > 1240:
            raise ValueError(f"Label outside canvas: {artist.get_text()}")
    bounds = corrected_artist.get_window_extent(renderer).transformed(ax.transData.inverted())
    if bounds.x0 < 555 or bounds.x1 > 805 or bounds.y0 < 940 or bounds.y1 > 970:
        raise ValueError("Corrected annotation exceeds the reference label area")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        path = output_dir / f"two_branch_adaptive_fusion_corrected.{extension}"
        fig.savefig(path, dpi=dpi, facecolor="white")
        paths.append(path)
    plt.close(fig)
    print("Verified: all labels inside canvas; corrected label fits original area")
    return paths


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    for output in render(args.output_dir, args.dpi):
        print(output)
