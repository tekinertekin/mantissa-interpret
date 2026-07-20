"""Occlusion sensitivity maps (Zeiler & Fergus, 2014).

Slide a small patch over the image, blank it out, and watch the target class
probability. Wherever hiding a region makes the class probability fall the
most, that region mattered most to the prediction. It needs nothing but the
model's forward pass, so it works for any classifier — no gradients, no
assumptions about the architecture.
"""
import numpy as np

__all__ = ["occlusion_map"]


def occlusion_map(model, image, target_class=None, patch=7, stride=3, fill=0.0):
    """Occlusion sensitivity heatmap for one image.

    Parameters
    ----------
    model : fitted ``mantissa_cnn.Sequential``
    image : array (C, H, W), float32
    target_class : int, optional
        Class to explain. ``None`` uses the model's own predicted class.
    patch : int
        Side length of the square occluding window.
    stride : int
        Step between window positions (smaller = finer + slower).
    fill : float
        Value written into the occluded window. ``0.0`` blacks it out; the
        image mean is a common neutral choice.

    Returns
    -------
    heat : array (H, W), float32 in [0, 1]
        Importance per pixel: 1 where occlusion hurt the target class most.
    """
    image = np.ascontiguousarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"image must be (C, H, W), got shape {image.shape}")
    C, H, W = image.shape
    if patch > H or patch > W:
        raise ValueError(f"patch {patch} larger than image {H}x{W}")

    base = model.predict_proba(image[None])[0]
    if target_class is None:
        target_class = int(base.argmax())
    base_p = float(base[target_class])

    # Window top-left positions, always including the bottom/right edge so no
    # border pixel is left un-probed.
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    if ys[-1] != H - patch:
        ys.append(H - patch)
    if xs[-1] != W - patch:
        xs.append(W - patch)

    # One forward pass over the whole batch of occluded copies.
    batch = np.repeat(image[None], len(ys) * len(xs), axis=0)
    coords = []
    k = 0
    for y in ys:
        for x in xs:
            batch[k, :, y:y + patch, x:x + patch] = fill
            coords.append((y, x))
            k += 1
    probs = model.predict_proba(batch)[:, target_class]
    drops = base_p - probs                       # how much hiding hurt the class

    # Average each patch's drop over the pixels it covered (overlaps blend).
    heat = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for (y, x), d in zip(coords, drops):
        heat[y:y + patch, x:x + patch] += d
        count[y:y + patch, x:x + patch] += 1.0
    heat /= np.maximum(count, 1.0)

    heat = np.maximum(heat, 0.0)                  # keep only positive importance
    peak = heat.max()
    if peak > 0:
        heat /= peak
    return heat
