import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers import FluxPipeline

from fora import CacheManager, FluxAdapter, FORAConfig
from fora.timing import timed_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Run FLUX image generation")
    parser.add_argument("--enable-fora", action="store_true")
    parser.add_argument("--model-path", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument('--prompt', default="A cat holding a sign that says hello world")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=3.5)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--interval', type=int, default=3)
    parser.add_argument('--output-path', default='outputs/flux.png')
    args = parser.parse_args()
    if min(args.interval, args.steps, args.height, args.width) < 1:
        parser.error("interval, steps, height, and width must be positive")
    return args


@torch.inference_mode()
def main():
    # parse arguments and load model
    args = parse_args()
    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to("cuda")
    pipe.transformer.eval()

    # apply monkeypatch
    adapter = nullcontext()
    if args.enable_fora:
        config = FORAConfig(args.interval)
        manager = CacheManager(config)
        adapter = FluxAdapter().patch(pipe, manager)

    model_memory = torch.cuda.memory_allocated(device="cuda") / 1e9
    with adapter:
        torch.cuda.reset_peak_memory_stats(device="cuda")
        image, elapsed = timed_inference(lambda: pipe(
            args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0])
    peak_memory = torch.cuda.max_memory_allocated(device="cuda") / 1e9

    # save image
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)

    print(f"Elapsed time: {elapsed:.2f} sec")
    print(f"Allocated after model load: {model_memory:.2f} GB")
    print(f"Peak allocated during inference: {peak_memory:.2f} GB")


if __name__ == '__main__':
    main()
