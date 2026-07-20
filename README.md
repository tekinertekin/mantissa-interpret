# mantissa-interpret

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)
[![Base](https://img.shields.io/badge/base-mantissa--cnn-4B8BBE.svg)](https://github.com/tekinertekin/mantissa-cnn)
[![Engine](https://img.shields.io/badge/engine-mantissa-00599C.svg)](https://github.com/tekinertekin/mantissa)

**Seeing what a CNN looks at.** A trained classifier gives you a label; it does
not tell you *why*. `mantissa-interpret` answers that with three classic
attribution methods that turn a prediction into a **heatmap over the input** —
bright where the pixels mattered for the class, dark where they did not.

Everything runs on a fitted [`mantissa_cnn.Sequential`](https://github.com/tekinertekin/mantissa-cnn)
model, using the model's **own forward and backward passes through the mantissa
C engine** — the same engine it was trained on. There is no PyTorch/TensorFlow
dependency: the heatmaps come out of the same low-precision C kernels that
produced the prediction.

## New to interpretability?

A CNN classifier maps an image to class scores. Two questions follow every
prediction:

- *Which pixels made it say "7"?* → **occlusion** and **saliency** give a
  per-pixel answer.
- *Which region did it focus on?* → **Grad-CAM** gives a per-region answer,
  and — unlike raw saliency — it is **class-discriminative** (ask it about "3"
  vs "7" on the same image and you get different maps).

These are the standard tools for debugging "right answer for the wrong reason"
failures (a model keying on a watermark, a background, an artifact).

## Install

```sh
pip install mantissa-interpret
```

Pulls in `mantissa-cnn` (and transitively `mantissa-nn` + the `mantissa-core`
engine). For plotting the heatmaps, `pip install mantissa-interpret[viz]`.

## The three methods

| method | cost | granularity | class-discriminative | needs |
|---|---|---|---|---|
| `occlusion_map` | forward only, O(patches) | coarse (patch) | yes | — |
| `saliency_map` | one backward pass | per-pixel (noisy) | weakly | input gradient |
| `grad_cam` | one backward pass | per-region (smooth) | **yes** | activations + their gradient at a conv layer |

Each takes a fitted model and a single `(C, H, W)` image and returns a 2-D
heatmap normalized to `[0, 1]`, aligned to the input.

<!-- Method details and worked math are filled in as each is added. -->

## Results

<!-- Heatmap galleries on trained mantissa-cnn models go here. -->

## License

MIT — Tekin Ertekin.
