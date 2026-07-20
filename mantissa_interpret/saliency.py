"""Vanilla gradient saliency (Simonyan, Vedaldi & Zisserman, 2014).

The first-order answer to "which input pixels would change this class score
the most?" is simply the gradient of that score with respect to the input.
We run one forward pass to prime the layers, seed the gradient of the target
logit, and backpropagate all the way to the pixels — using the same layer
``backward`` passes the model trains with, only carried one step further (to
the input instead of stopping at the first layer's weights).
"""
import numpy as np

__all__ = ["saliency_map"]


def saliency_map(model, image, target_class=None):
    """Gradient-magnitude saliency heatmap for one image.

    Parameters
    ----------
    model : fitted ``mantissa_cnn.Sequential``
    image : array (C, H, W), float32
    target_class : int, optional
        Class whose score to differentiate. ``None`` uses the predicted class.

    Returns
    -------
    heat : array (H, W), float32 in [0, 1]
        Per-pixel ``|d score_c / d pixel|``, reduced over channels by max.

    Notes
    -----
    Differentiates the raw class *logit* (pre-softmax), the standard choice:
    it isolates evidence for the class without the softmax coupling every class
    into the gradient.
    """
    image = np.ascontiguousarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"image must be (C, H, W), got shape {image.shape}")
    backend = model._backend

    # Forward pass: each layer caches what its backward needs.
    h = image[None]
    for layer in model.layers:
        h = layer.forward(h, backend)
    logits = h                                    # (1, n_classes)

    if target_class is None:
        target_class = int(logits[0].argmax())

    # Seed: d(logit_c) / d(logits) is the one-hot vector for c.
    dY = np.zeros_like(logits)
    dY[0, target_class] = 1.0

    # Backprop to the input. need_dx=True on every layer (including the first,
    # unlike training which stops at layer 0) so we get d(logit_c)/d(input).
    grad = dY
    for layer in reversed(model.layers):
        grad = layer.backward(grad, backend, need_dx=True)
    grad = np.asarray(grad)[0]                     # (C, H, W)

    sal = np.abs(grad).max(axis=0)                 # (H, W)
    peak = sal.max()
    if peak > 0:
        sal = sal / peak
    return sal.astype(np.float32)
