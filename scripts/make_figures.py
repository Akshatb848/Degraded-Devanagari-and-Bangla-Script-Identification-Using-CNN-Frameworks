"""Redraw the README training-curve chart from the Keras log saved in the notebook.

The notebook contains no plots; this chart is drawn only from the per-epoch
accuracies it printed.  Nothing is re-trained.  Usage:  python scripts/make_figures.py
"""
import json
import pathlib
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9c8c3",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e7e6e2", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlesize": 13,
})


def stream_text(notebook, cell_index):
    nb = json.loads((ROOT / notebook).read_text(encoding="utf-8"))
    outs = nb["cells"][cell_index].get("outputs", [])
    text = "".join("".join(o.get("text", "")) for o in outs if o.get("output_type") == "stream")
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def keras_epochs(text):
    """(train_acc, val_acc) per epoch from a Keras progress log."""
    pairs = re.findall(r"- accuracy: ([0-9.]+) - loss: [0-9.e+-]+ - val_accuracy: ([0-9.]+)", text)
    return [(float(a), float(b)) for a, b in pairs]


def curve(ep, fname, title, caption, val_label, best=None):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    x = range(1, len(ep) + 1)
    ax.plot(x, [a for a, _ in ep], color=BLUE, lw=2, marker="o", ms=4, label="Training accuracy")
    ax.plot(x, [b for _, b in ep], color=ORANGE, lw=2, marker="o", ms=4, label=val_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, loc="left", pad=10)
    ax.legend(frameon=False, loc="lower right")
    if best:
        ax.annotate(best[2], xy=best[:2], xytext=(best[0] + 0.4, 0.45), fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    fig.text(0.01, -0.02, textwrap.fill(caption, 110), fontsize=8.5, color=INK2, va="top")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    return OUT / fname


def training_curve():
    log = stream_text("Degraded_Script_Classifier.ipynb", 9)
    test_acc = float(re.search(r"Test accuracy: ([0-9.]+)", log).group(1))
    ep = keras_epochs(log)
    best_epoch = max(range(len(ep)), key=lambda i: ep[i][1]) + 1
    return curve(
        ep, "dsc_training_curve.png",
        "Custom CNN (augmented run, early stopping): accuracy per epoch",
        'Plotted from the Keras log saved in Degraded_Script_Classifier.ipynb, cell 10. '
        '"Validation" reads the same 7,628 images as training.',
        "Validation accuracy (same folder, no augmentation)",
        best=(best_epoch, ep[best_epoch - 1][1],
              f'epoch {best_epoch}: {test_acc:.4f}, weights restored\n(reported "Test accuracy")'),
    )


if __name__ == "__main__":
    print("written", training_curve())
