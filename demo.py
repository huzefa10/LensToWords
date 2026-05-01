"""
demo.py — command-line demo for LensToWords

Usage:
    python demo.py --image path/to/image.jpg
    python demo.py --image path/to/image.jpg --weights weights/
    python demo.py --image path/to/image.jpg --beam

Weights directory must contain:
    best_perciever.pth
    best_gpt2.pth
    best_cross_attention.pth

Download weights: https://huggingface.co/huzefa10/lenstowords (see README)
"""

import argparse
import os
import sys
import time

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')


def parse_args():
    p = argparse.ArgumentParser(description='Generate a caption for an image.')
    p.add_argument('--image',   required=True,  help='Path to input image')
    p.add_argument('--weights', default=WEIGHTS_DIR,
                   help='Directory containing best_perciever.pth, best_gpt2.pth, '
                        'best_cross_attention.pth')
    p.add_argument('--beam',    action='store_true',
                   help='Use beam search instead of top-k sampling')
    p.add_argument('--top-k',   type=int,   default=50,  help='Top-k for sampling (default 50)')
    p.add_argument('--temp',    type=float, default=1.0, help='Sampling temperature (default 1.0)')
    p.add_argument('--beam-width', type=int, default=5,  help='Beam width (default 5)')
    p.add_argument('--device',  default=None, help='cuda or cpu (auto-detected if omitted)')
    return p.parse_args()


def check_weights(weights_dir):
    required = ['best_perciever.pth', 'best_gpt2.pth', 'best_cross_attention.pth']
    missing  = [f for f in required if not os.path.isfile(os.path.join(weights_dir, f))]
    if missing:
        print(f'[error] Missing weight files in {weights_dir}:')
        for f in missing:
            print(f'          {f}')
        print('\nDownload from: https://huggingface.co/huzefa10/lenstowords')
        sys.exit(1)


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f'[error] Image not found: {args.image}')
        sys.exit(1)

    check_weights(args.weights)

    print('Loading model...', end=' ', flush=True)
    t0 = time.time()

    from src.infer import build_model, generate_caption, generate_caption_beam

    device = args.device
    model, encoder = build_model(
        perciever_path   = os.path.join(args.weights, 'best_perciever.pth'),
        gpt2_blocks_path = os.path.join(args.weights, 'best_gpt2.pth'),
        cross_attn_path  = os.path.join(args.weights, 'best_cross_attention.pth'),
        device           = device or ('cuda' if __import__('torch').cuda.is_available() else 'cpu'),
    )
    print(f'done ({time.time() - t0:.1f}s)')

    print(f'Image : {args.image}')
    print('Generating caption...', end=' ', flush=True)
    t1 = time.time()

    if args.beam:
        caption = generate_caption_beam(
            model, encoder, args.image,
            beam_width=args.beam_width,
            device=model.perciever.latents.device,
        )
        mode = f'beam (width={args.beam_width})'
    else:
        caption = generate_caption(
            model, encoder, args.image,
            top_k=args.top_k,
            temperature=args.temp,
            device=model.perciever.latents.device,
        )
        mode = f'top-k (k={args.top_k}, temp={args.temp})'

    print(f'done ({time.time() - t1:.1f}s)')
    print(f'\nCaption [{mode}]:\n  {caption}')


if __name__ == '__main__':
    main()
