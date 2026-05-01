# LensToWords

An image captioning system built from scratch — evolving from a TensorFlow LSTM baseline to a PyTorch Perceiver + GPT-2 model fine-tuned on COCO and Instagram.

Architecture diagram coming soon

---

## Project overview

This project documents a full, iterative build of an image captioning pipeline across 6 architectures. Each iteration introduced a specific hypothesis, tested it, and informed the next design. All training was done on Kaggle (2× T4 GPU).

| Version | Architecture | Dataset | BLEU-4 |
|---|---|---|---|
| v1 | InceptionV3 + LSTM + Bahdanau attention | Flickr30k | 0.0674 |
| v2 | InceptionV3 + multi-head attention | Instagram (~21k) | — (Instagram eval) |
| v3a | ConvNeXt-Tiny (multi-scale) + custom Transformer | Flickr30k | Phase 1 baseline |
| v3b | ConvNeXt-Tiny (multi-scale) + Perceiver + GPT-2 | Flickr30k | **0.0656** (Phase 2d) |
| v3c | Same as v3b, COCO fine-tuned | COCO 118k | **0.1341** |
| v3d | Same as v3b, Instagram fine-tuned | CLIP-filtered Instagram | 0.0711 (recovery) |

---

## Architecture (final — v3b onward)

```
Image (224×224)
    │
    ▼
ConvNeXt-Tiny (frozen)
    │  spatial feature map: (B, 49, 1024)
    ▼
Perceiver
    │  learnable latents cross-attend to image features
    │  output: (B, 64, 768)
    ▼
GPT-2 (gpt2-small, 12 layers)
    │  cross-attention injected after every block
    │  loss: CrossEntropyLoss(ignore_index=50256)
    ▼
Caption tokens  →  decoded text
```

**Key design decisions:**
- ConvNeXt-Tiny pre-computed features stored in LMDB (6 shards) — avoids re-encoding every epoch
- Perceiver compresses 49 spatial tokens to 64 learnable latents before GPT-2
- Cross-attention injected into GPT-2 rather than prepend — cleaner separation of vision and language streams
- Gradient checkpointing enabled in `__init__` before extracting component references (shared Python objects)
- `MAX_LEN = 64`, `BATCH_SIZE = 32` (reduced from 64 after GPU OOM in Phase 2b)

---

## Training progression

### v3b — Flickr30k (Phases 2a–2d)

| Phase | Setup | Epochs | LR | BLEU-4 |
|---|---|---|---|---|
| 2a | Perceiver only, GPT-2 frozen, prepend | 25 | 1e-4 | — |
| 2b | Perceiver + GPT-2 jointly, prepend | 25 | P: 1e-5 / G: 1e-6 | 0.0336 |
| 2c | CrossAttention only, rest frozen | 25 | 1e-4 | — |
| 2d | All components, injection architecture | 25 | 1e-5 | **0.0656** |

Phase 2d loads Perceiver + GPT-2 weights from 2b and CrossAttention from 2c.

### v3c — COCO fine-tuning

| Phase | Data | BLEU-4 |
|---|---|---|
| 3a | COCO 50k subset | 0.1331 |
| 3a extended | Full COCO 118k (continuing from 3a weights) | **0.1341** |

### v3d — Instagram fine-tuning

CLIP (ViT-B/32) cosine > 0.27 filter retained **4,022 samples** from the raw Instagram dataset — only 11.5%, confirming how noisy Instagram captions are. All BLEU evaluation in v3d is on **Flickr30k val** (3,179 images, 5 references each).

| Phase | Setup | BLEU-4 (Flickr val) |
|---|---|---|
| 3b | Instagram only, 5 epochs | 0.0448 |
| Recovery 1 | Mixed Instagram + Flickr | **0.0711** |
| 3c | Instagram only, full run | 0.0633 |

Instagram fine-tuning consistently degraded Flickr BLEU — domain gap is significant.

---

## Repository structure

```
LensToWords/
├── notebooks/
│   ├── v1_inceptionv3_lstm.ipynb
│   ├── v2_inceptionv3_attention.ipynb
│   ├── v3a_convnext_transformer.ipynb
│   ├── v3b_perceiver_gpt2.ipynb
│   ├── v3c_coco_finetuning.ipynb
│   └── v3d_instagram_finetuning.ipynb
├── src/
│   ├── model.py        # PercieverGptDecoder architecture
│   └── infer.py        # caption generation (top-k + beam search)
├── assets/
│   └── architecture_evolution.png
├── demo.py             # command-line demo
└── weights/            # place downloaded .pth files here (see below)
```

---

## Quickstart

**1. Install dependencies**

```bash
pip install torch torchvision transformers pillow
```

**2. Download weights**

Place the following files in `weights/`:
- `best_perciever.pth`
- `best_gpt2.pth`
- `best_cross_attention.pth`

> Weights available at: [Kaggle dataset — huzefamerchant/lenstowords-weights] *(link to be added)*

**3. Run demo**

```bash
# Top-k sampling (default)
python demo.py --image path/to/image.jpg

# Beam search
python demo.py --image path/to/image.jpg --beam --beam-width 5

# Custom temperature
python demo.py --image path/to/image.jpg --top-k 30 --temp 0.8
```

**4. Use in code**

```python
from src.infer import build_model, generate_caption

model, encoder = build_model(
    perciever_path   = 'weights/best_perciever.pth',
    gpt2_blocks_path = 'weights/best_gpt2.pth',
    cross_attn_path  = 'weights/best_cross_attention.pth',
)

caption = generate_caption(model, encoder, 'path/to/image.jpg')
print(caption)
```

---

## Notebooks

Each notebook is self-contained and documents one architecture. They are written to be read sequentially — each one starts from where the previous left off and explains the reasoning behind the change.

| Notebook | What it documents |
|---|---|
| `v1_inceptionv3_lstm.ipynb` | Baseline: LSTM decoder with Bahdanau attention |
| `v2_inceptionv3_attention.ipynb` | Switch to multi-head self-attention; Instagram dataset exploration |
| `v3a_convnext_transformer.ipynb` | Move to PyTorch; ConvNeXt encoder; custom Transformer decoder |
| `v3b_perceiver_gpt2.ipynb` | Perceiver + GPT-2 with cross-attention injection; full 4-phase training |
| `v3c_coco_finetuning.ipynb` | COCO fine-tuning in two phases (50k → 118k) |
| `v3d_instagram_finetuning.ipynb` | CLIP-filtered Instagram fine-tuning; domain gap analysis |

---

## Key findings

- **LSTM → Transformer:** Attention gives better spatial grounding but LSTM trains faster with less data.
- **Custom Transformer → GPT-2:** Pre-trained language model provides far better fluency and generalisation.
- **Prepend → Cross-attention injection:** Injection (Phase 2d) outperforms prepend (Phase 2b) — 0.0656 vs 0.0336 BLEU-4 on Flickr val.
- **COCO scale matters:** 50k → 118k produced only marginal gain (0.1331 → 0.1341), suggesting diminishing returns at this data scale for this architecture.
- **Instagram domain gap:** CLIP filtering retained only 11.5% of captions. Fine-tuning on Instagram alone consistently degraded Flickr BLEU — domain gap is too large without mixing.

---

## Tech stack

`PyTorch` · `HuggingFace Transformers` · `TorchVision` · `TensorFlow/Keras (v1–v2)` · `LMDB` · `NLTK BLEU` · `CLIP (ViT-B/32)` · `Kaggle (2× T4 GPU)`

---

## Author

**Huzefa Merchant**  
[LinkedIn](https://linkedin.com/in/huzefa-merchant) · [GitHub](https://github.com/huzefa10)
