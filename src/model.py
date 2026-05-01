"""
src/model.py — PercieverGptDecoder: final captioning architecture (v3b onward)

Pipeline:
  ConvNextMultiScale  →  multi-scale ConvNeXt-Tiny features  (B, 64, 768)
  Perceiver           →  cross-attention to latents           (B, 16, 768)
  GPT2xAttn           →  GPT-2 + cross-attention inject       (B, T, vocab_size)

Weights are saved per-component during training; use load_weights() to restore.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

# ── Constants ─────────────────────────────────────────────────────────────────

IMG_DIM    = 768    # ConvNextMultiScale concatenated output (3 stages × 256)
N_LATENTS  = 16     # Perceiver latent tokens
D_MODEL    = 768    # GPT-2 hidden size
N_HEADS_P  = 8      # Perceiver cross-attention heads
N_HEADS_G  = 12     # GPT-2 cross-attention heads
PAD_ID     = 50256  # GPT-2 EOS token reused as padding


# ── Image encoder ─────────────────────────────────────────────────────────────

class ConvNextMultiScale(nn.Module):
    """
    Frozen ConvNeXt-Tiny backbone with multi-scale extraction (timm).
    Extracts stages 1, 2, 3 (channels: 192, 384, 768), pools each to 8×8,
    projects each to 256-d, then concatenates along the feature axis:
        (B, 64, 256) × 3  →  cat(dim=2)  →  (B, 64, 768)
    Used at inference time; training used pre-computed LMDB embeddings.
    """
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            'convnext_tiny', pretrained=True, features_only=True, out_indices=(1, 2, 3)
        )
        self.pool  = nn.AdaptiveAvgPool2d((8, 8))
        self.proj1 = nn.Linear(192, 256)
        self.proj2 = nn.Linear(384, 256)
        self.proj3 = nn.Linear(768, 256)
        self.norm  = nn.LayerNorm(768)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x: (B, 3, 224, 224)
        s1, s2, s3 = self.backbone(x)      # (B,192,H1,W1), (B,384,H2,W2), (B,768,H3,W3)

        def pool_proj(feat, proj):
            f = self.pool(feat)             # (B, C, 8, 8)
            B, C, H, W = f.shape
            f = f.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, 64, C)
            return proj(f)                  # (B, 64, 256)

        f1 = pool_proj(s1, self.proj1)     # (B, 64, 256)
        f2 = pool_proj(s2, self.proj2)     # (B, 64, 256)
        f3 = pool_proj(s3, self.proj3)     # (B, 64, 256)

        return self.norm(torch.cat([f1, f2, f3], dim=2))   # (B, 64, 768)


# ── Perceiver ─────────────────────────────────────────────────────────────────

class Perceiver(nn.Module):
    """
    Cross-attention from learnable latents to image features.
    Compresses variable-length spatial features into a fixed latent sequence.
    """
    def __init__(self, img_dim=IMG_DIM, n_latents=N_LATENTS, d_model=D_MODEL, n_heads=N_HEADS_P):
        super().__init__()
        self.latents   = nn.Parameter(torch.randn(n_latents, d_model))
        self.img_proj  = nn.Linear(img_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1     = nn.LayerNorm(d_model)
        self.ff        = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, img_feats):
        # img_feats: (B, S, img_dim)
        kv = self.img_proj(img_feats)                              # (B, S, d_model)
        B  = kv.size(0)
        q  = self.latents.unsqueeze(0).expand(B, -1, -1)          # (B, n_latents, d_model)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x  = self.norm1(q + attn_out)
        x  = self.norm2(x + self.ff(x))
        return x                                                   # (B, n_latents, d_model)


# ── Cross-attention injected into each GPT-2 block ───────────────────────────

class CrossAttention(nn.Module):
    """Attends each decoder token to Perceiver latents; injected after every GPT-2 block."""
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS_G):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, context):
        # x: (B, T, d_model)  |  context: (B, n_latents, d_model)
        out, _ = self.attn(x, context, context)
        return self.norm(x + out)


# ── GPT-2 with cross-attention injection ─────────────────────────────────────

class GPT2withCrossAttention(nn.Module):
    """
    GPT-2 (gpt2-small) with a shared CrossAttention module injected after every block.
    gradient_checkpointing_enable() is called before extracting component references
    so that checkpointing propagates through the shared Python objects.
    """
    def __init__(self):
        super().__init__()
        gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2.gradient_checkpointing_enable()    # must precede reference extraction
        self.wte         = gpt2.transformer.wte
        self.wpe         = gpt2.transformer.wpe
        self.drop        = gpt2.transformer.drop
        self.gpt2_blocks = gpt2.transformer.h   # ModuleList, 12 blocks
        self.ln_f        = gpt2.transformer.ln_f
        self.lm_head     = gpt2.lm_head
        self.cross_attn  = CrossAttention()

    def forward(self, input_ids, context):
        # input_ids: (B, T)  |  context: (B, n_latents, d_model)
        B, T  = input_ids.shape
        pos   = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x     = self.drop(self.wte(input_ids) + self.wpe(pos))   # (B, T, 768)
        for block in self.gpt2_blocks:
            x = block(x)[0]                     # causal self-attention
            x = self.cross_attn(x, context)     # cross-attention to image latents
        x = self.ln_f(x)
        return self.lm_head(x)                  # (B, T, 50257)


# ── Full decoder ──────────────────────────────────────────────────────────────

class PercieverGptDecoder(nn.Module):
    """
    Full captioning model: Perceiver + GPT-2 with cross-attention injection.

    Training results:
      Flickr30k val  (Phase 2d)  BLEU-4: 0.0656
      COCO 50k       (Phase 3a)  BLEU-4: 0.1331
      COCO 118k      (Phase 3a+) BLEU-4: 0.1341
      Flickr val     (Recovery)  BLEU-4: 0.0711
    """
    def __init__(self, img_dim=IMG_DIM, n_latents=N_LATENTS):
        super().__init__()
        self.perciever = Perceiver(img_dim=img_dim, n_latents=n_latents)
        self.gpt2      = GPT2withCrossAttention()

    def forward(self, img_feats, input_ids):
        # img_feats: (B, S, img_dim)  |  input_ids: (B, T)
        context = self.perciever(img_feats)     # (B, n_latents, 768)
        logits  = self.gpt2(input_ids, context) # (B, T, 50257)
        return logits


# ── Weight loading ────────────────────────────────────────────────────────────

def load_weights(model, perciever_path, gpt2_blocks_path, cross_attn_path, device='cpu'):
    """
    Restore per-component weights saved during training.

    Weights were saved as:
        torch.save(decoder_model.module.perciever.state_dict(),   'best_perciever.pth')
        torch.save(decoder_model.module.gpt2.gpt2_blocks.state_dict(), 'best_gpt2.pth')
        torch.save(decoder_model.module.gpt2.cross_attn.state_dict(),  'best_cross_attention.pth')

    If loading a full DataParallel checkpoint instead, strip the prefix first:
        sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    """
    kw = dict(map_location=device, weights_only=True)
    model.perciever.load_state_dict(torch.load(perciever_path,   **kw))
    model.gpt2.gpt2_blocks.load_state_dict(torch.load(gpt2_blocks_path, **kw))
    model.gpt2.cross_attn.load_state_dict(torch.load(cross_attn_path,   **kw))
    model.eval()
    return model
