"""mantissa-interpret: CNN interpretability on the mantissa C engine.

Attribution / visualization methods that reveal *what a trained
``mantissa_cnn.Sequential`` model looks at* when it makes a prediction. Each
method takes a fitted model and a single NCHW image and returns a 2-D heatmap
aligned to that image — computed with the model's own forward (and, where
needed, backward) passes, so it runs through the exact C engine the model was
trained on. No deep-learning framework is involved.

Three methods, cheapest first:

- ``occlusion_map``  — forward-only: slide a gray patch over the image and see
  how much the target class probability drops. Model-agnostic, no gradients.
- ``saliency_map``   — one backward pass: |gradient of the class score w.r.t.
  each input pixel|. Fine-grained but noisy.
- ``grad_cam``       — Grad-CAM: gradient-weighted activation map at a chosen
  convolutional layer. Coarse but class-discriminative and low-noise.

>>> from mantissa_cnn import models, datasets
>>> from mantissa_interpret import occlusion_map, saliency_map, grad_cam
>>> net = models.lenet5(); net.fit(X, y, epochs=3)
>>> heat = grad_cam(net, image, target_class=7)   # (H, W) in [0, 1]
"""
__version__ = "0.1.0"
__all__ = []
