"""Grad-CAM (Selvaraju et al., 2017).

Saliency is per-pixel and noisy; Grad-CAM is per-region and class-
discriminative. It works at a convolutional layer, where each channel is a
learned feature detector over a coarse spatial grid. For a target class we
weight every channel by how much its activation *increasing* would raise the
class score (the average gradient), sum the channels with those weights, and
keep the positive part. The result is a coarse map of "where the evidence for
this class lives", which we upsample back to the image.

We reuse exactly the forward/backward the model trains with: capture the target
conv layer's output on the way up, backpropagate the target logit down to that
same layer to get its gradient, and combine.
"""
import numpy as np
from mantissa_cnn.layers import Conv2D

__all__ = ["grad_cam"]


def _bilinear(cam, out_hw):
    """Bilinearly resize a 2-D map to (H, W). Dependency-free."""
    H, W = out_hw
    h, w = cam.shape
    if (h, w) == (H, W):
        return cam.astype(np.float32)
    ys = np.linspace(0.0, h - 1, H)
    xs = np.linspace(0.0, w - 1, W)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None].astype(np.float32)
    wx = (xs - x0)[None, :].astype(np.float32)
    top = cam[y0][:, x0] * (1 - wx) + cam[y0][:, x1] * wx
    bot = cam[y1][:, x0] * (1 - wx) + cam[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def grad_cam(model, image, target_class=None, target_layer=None):
    """Grad-CAM heatmap for one image.

    Parameters
    ----------
    model : fitted ``mantissa_cnn.Sequential``
    image : array (C, H, W), float32
    target_class : int, optional
        Class to explain. ``None`` uses the predicted class.
    target_layer : int or Conv2D, optional
        Which convolutional layer to read. ``None`` uses the *last* Conv2D
        (the standard, most class-specific choice).

    Returns
    -------
    heat : array (H, W), float32 in [0, 1]
    """
    image = np.ascontiguousarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"image must be (C, H, W), got shape {image.shape}")
    C, H, W = image.shape
    backend = model._backend
    layers = model.layers

    if target_layer is None:
        conv_idx = [i for i, l in enumerate(layers) if isinstance(l, Conv2D)]
        if not conv_idx:
            raise ValueError("model has no Conv2D layer to run Grad-CAM on")
        target_idx = conv_idx[-1]
    elif isinstance(target_layer, int):
        target_idx = target_layer
    else:
        target_idx = layers.index(target_layer)

    # Forward, copying out the target layer's activation A (the engine reuses
    # buffers in place, so we must copy, not alias).
    A = None
    h = image[None]
    for i, layer in enumerate(layers):
        h = layer.forward(h, backend)
        if i == target_idx:
            A = np.array(h)                       # (1, K, h', w')
    logits = h

    if target_class is None:
        target_class = int(logits[0].argmax())

    dY = np.zeros_like(logits)
    dY[0, target_class] = 1.0

    # Backprop only down to the layer just above the target: the gradient
    # handed into that layer's backward is d(logit_c)/d(A).
    grad = dY
    for i in range(len(layers) - 1, target_idx, -1):
        grad = layers[i].backward(grad, backend, need_dx=True)
    dA = np.array(grad)                            # (1, K, h', w')

    alpha = dA.mean(axis=(2, 3))                   # (1, K) channel importances
    cam = np.maximum((alpha[:, :, None, None] * A).sum(axis=1)[0], 0.0)  # (h', w')

    cam = _bilinear(cam, (H, W))
    peak = cam.max()
    if peak > 0:
        cam = cam / peak
    return cam.astype(np.float32)
