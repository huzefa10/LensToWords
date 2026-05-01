"""
src/infer.py — inference utilities for PercieverGptDecoder

Usage:
    from src.model import PercieverGptDecoder, ConvNextMultiScale, load_weights
    from src.infer import build_model, generate_caption

    model = build_model(
        perciever_path   = 'weights/best_perciever.pth',
        gpt2_blocks_path = 'weights/best_gpt2.pth',
        cross_attn_path  = 'weights/best_cross_attention.pth',
    )
    caption = generate_caption(model, 'path/to/image.jpg')
    print(caption)
"""

import re
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import GPT2Tokenizer

from src.model import PercieverGptDecoder, ConvNextMultiScale, load_weights

# ── Setup ─────────────────────────────────────────────────────────────────────

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(perciever_path, gpt2_blocks_path, cross_attn_path, device=DEVICE):
    """Load architecture + weights; return eval-mode model and image encoder."""
    model   = PercieverGptDecoder().to(device)
    encoder = ConvNextMultiScale().to(device)
    load_weights(model, perciever_path, gpt2_blocks_path, cross_attn_path, device=device)
    encoder.eval()
    return model, encoder


# ── Image preprocessing ───────────────────────────────────────────────────────

def load_image(path):
    """Open an image file and return a preprocessed (1, 3, 224, 224) tensor."""
    img = Image.open(path).convert('RGB')
    return IMG_TRANSFORM(img).unsqueeze(0)


# ── Caption generation ────────────────────────────────────────────────────────

def _clean(text):
    """Strip excess whitespace and normalise punctuation spacing."""
    text = text.strip()
    text = re.sub(r'\s([?.!,])', r'\1', text)
    return text


def generate_caption(model, encoder, image_path, max_len=64, top_k=50,
                     temperature=1.0, device=DEVICE):
    """
    Generate a caption for a single image using top-k sampling.

    Args:
        model       : PercieverGptDecoder (loaded with load_weights)
        encoder     : ImageEncoder (frozen ConvNeXt-Base)
        image_path  : str — path to image file
        max_len     : int — maximum tokens to generate (default 64, matches training)
        top_k       : int — top-k candidates at each step
        temperature : float — softmax temperature (1.0 = no scaling)
        device      : str — 'cuda' or 'cpu'

    Returns:
        str — generated caption
    """
    tokenizer = get_tokenizer()
    pad_id    = tokenizer.eos_token_id   # 50256

    img_tensor = load_image(image_path).to(device)

    with torch.no_grad():
        img_feats  = encoder(img_tensor)                  # (1, 49, 1024)
        context    = model.perciever(img_feats)            # (1, 64, 768)

        input_ids  = torch.tensor([[pad_id]], device=device)   # start with EOS/BOS

        for _ in range(max_len - 1):
            logits     = model.gpt2(input_ids, context)   # (1, T, 50257)
            next_logits = logits[0, -1] / temperature      # (50257,)

            # top-k filtering
            topk_vals, topk_idx = torch.topk(next_logits, top_k)
            probs = torch.softmax(topk_vals, dim=-1)
            next_token = topk_idx[torch.multinomial(probs, 1)]

            if next_token.item() == pad_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # drop the leading BOS token before decoding
        tokens  = input_ids[0, 1:].tolist()
        caption = tokenizer.decode(tokens, skip_special_tokens=True)

    return _clean(caption)


def generate_caption_beam(model, encoder, image_path, max_len=64, beam_width=5,
                          device=DEVICE):
    """
    Greedy beam search (no external library dependency).

    Returns the highest-scoring beam as a string.
    """
    tokenizer = get_tokenizer()
    pad_id    = tokenizer.eos_token_id

    img_tensor = load_image(image_path).to(device)

    with torch.no_grad():
        img_feats = encoder(img_tensor)           # (1, 49, 1024)
        context   = model.perciever(img_feats)    # (1, 64, 768)

        # each beam: (log_prob, token_list)
        beams = [(0.0, [pad_id])]
        completed = []

        for _ in range(max_len - 1):
            candidates = []
            for log_prob, tokens in beams:
                ids    = torch.tensor([tokens], device=device)
                logits = model.gpt2(ids, context)[0, -1]       # (50257,)
                log_p  = torch.log_softmax(logits, dim=-1)
                topk_lp, topk_idx = torch.topk(log_p, beam_width)
                for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                    candidates.append((log_prob + lp, tokens + [idx]))

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = []
            for lp, toks in candidates[:beam_width]:
                if toks[-1] == pad_id:
                    completed.append((lp, toks))
                else:
                    beams.append((lp, toks))
            if not beams:
                break

        if not completed:
            completed = beams
        best_tokens = max(completed, key=lambda x: x[0])[1]
        caption = tokenizer.decode(best_tokens[1:], skip_special_tokens=True)

    return _clean(caption)
