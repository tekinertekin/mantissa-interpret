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

    # "How occlusion works" — the mechanism. Covering the *right* region must
    # collapse P(class) while covering empty space must not. MNIST digits are
    # robust to a tiny patch, so we use a bigger window and pick, across a pool
    # of test images, the patch that causes the largest real probability drop.
    from matplotlib.patches import Rectangle
    P = 12

    def _occ_probs(im, cls):
        C, H, W = im.shape
        ys = list(range(0, H - P + 1, 2))
        xs = list(range(0, W - P + 1, 2))
        batch = np.repeat(im[None], len(ys) * len(xs), axis=0)
        coords, k = [], 0
        for y in ys:
            for x in xs:
                batch[k, :, y:y + P, x:x + P] = 0.0
                coords.append((y, x)); k += 1
        return coords, net.predict_proba(batch)[:, cls]

    best = None
    for i in range(min(60, len(Xte))):
        c = int(net.predict(Xte[i][None])[0])
        if c != int(yte[i]):
            continue
        bp = float(net.predict_proba(Xte[i][None])[0, c])
        coords, probs = _occ_probs(Xte[i], c)
        drop = bp - float(probs.min())
        if best is None or drop > best[0]:
            best = (drop, i, c, bp, coords, probs)
    _, i7, cls, base_p, coords, probs = best
    img = Xte[i7]
    (y_i, x_i), p_i = coords[int(np.argmin(probs))], float(probs.min())   # biggest drop
    (y_b, x_b), p_b = coords[int(np.argmax(probs))], float(probs.max())   # no drop
    im_i = img.copy(); im_i[:, y_i:y_i + P, x_i:x_i + P] = 0.0
    im_b = img.copy(); im_b[:, y_b:y_b + P, x_b:x_b + P] = 0.0
    heat = occlusion_map(net, img, target_class=cls, patch=P, stride=2)
    fig3, ax3 = plt.subplots(1, 4, figsize=(8.5, 2.4))
    _overlay(ax3[0], img[0], None, f"input\nP({cls}) = {base_p:.2f}")
    _overlay(ax3[1], im_i[0], None, f"cover a stroke\nP({cls}) = {p_i:.2f}")
    ax3[1].add_patch(Rectangle((x_i - .5, y_i - .5), P, P, fill=False, ec="lime", lw=1.5))
    _overlay(ax3[2], im_b[0], None, f"cover empty space\nP({cls}) = {p_b:.2f}")
    ax3[2].add_patch(Rectangle((x_b - .5, y_b - .5), P, P, fill=False, ec="cyan", lw=1.5))
    _overlay(ax3[3], img[0], heat, "importance map")
    fig3.suptitle("How occlusion works: hide a region, watch the class probability", fontsize=11)
    fig3.tight_layout()
    out3 = os.path.join(os.path.dirname(__file__), os.pardir, "assets", "how_occlusion_works.png")
    fig3.savefig(out3, dpi=130, bbox_inches="tight")
    print("saved", out3)


if __name__ == "__main__":
    main()
