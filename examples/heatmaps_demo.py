"""Generate the README heatmap gallery.

Trains a small LeNet-5 on an MNIST subset with the mantissa C engine, then runs
all three attribution methods on a few test digits and saves a side-by-side
comparison to ``assets/heatmaps.png``. Also shows Grad-CAM's class-
discriminativeness (same image, two target classes) in ``assets/gradcam_classes.png``.

    python examples/heatmaps_demo.py
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MANTISSA_CNN_DATA",
                      os.path.join(os.path.dirname(__file__), os.pardir, "data"))

import numpy as np
import matplotlib.pyplot as plt

from mantissa_cnn import models, datasets
from mantissa_interpret import occlusion_map, saliency_map, grad_cam


def _overlay(ax, base, heat=None, title=None):
    ax.imshow(base, cmap="gray")
    if heat is not None:
        ax.imshow(heat, cmap="jet", alpha=0.5)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    datasets.download("mnist")                       # idempotent
    Xtr, ytr, Xte, yte = datasets.subset("mnist", n_train=8000, n_test=2000, seed=0)

    net = models.lenet5(seed=0)
    net.fit(Xtr, ytr, epochs=3, batch_size=64, lr=0.05, verbose=True)
    print(f"test accuracy (2k): {net.score(Xte, yte):.3f}")

    # a few varied, correctly-classified digits
    preds = net.predict(Xte)
    picks, seen = [], set()
    for i in range(len(Xte)):
        d = int(yte[i])
        if preds[i] == d and d not in seen:
            picks.append(i); seen.add(d)
        if len(picks) == 4:
            break

    methods = [("occlusion", occlusion_map), ("saliency", saliency_map), ("Grad-CAM", grad_cam)]
    n = len(picks)
    fig, axes = plt.subplots(n, 4, figsize=(7.5, 1.9 * n))
    for r, i in enumerate(picks):
        img = Xte[i]                                 # (1, 28, 28)
        pred = int(net.predict(img[None])[0])
        _overlay(axes[r, 0], img[0], None, f"input → {pred}")
        for c, (name, fn) in enumerate(methods, start=1):
            heat = fn(net, img, target_class=pred)
            _overlay(axes[r, c], img[0], heat, name if r == 0 else None)
    fig.suptitle("mantissa-interpret — what the CNN looks at (MNIST, LeNet-5)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(__file__), os.pardir, "assets"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), os.pardir, "assets", "heatmaps.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)

    # Grad-CAM is class-discriminative: same image, different target classes.
    img = Xte[picks[0]]
    true = int(yte[picks[0]])
    others = [c for c in range(10) if c != true][:2]
    fig2, ax2 = plt.subplots(1, 3, figsize=(6, 2.2))
    _overlay(ax2[0], img[0], None, f"input (true {true})")
    _overlay(ax2[1], img[0], grad_cam(net, img, target_class=true), f"Grad-CAM · class {true}")
    _overlay(ax2[2], img[0], grad_cam(net, img, target_class=others[0]), f"Grad-CAM · class {others[0]}")
    fig2.suptitle("Grad-CAM is class-discriminative", fontsize=11)
    fig2.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), os.pardir, "assets", "gradcam_classes.png")
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print("saved", out2)


if __name__ == "__main__":
    main()
