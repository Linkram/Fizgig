"""MiniMax H3 RefMod — make a reference mod from the dataset and optimise it against the
frozen H3 base (see fizgig.minimax.refmod). The GUI's "MiniMax H3 RefMod" Base Model entry
builds this command; the caches come from the ordinary minimax_cache_latents /
minimax_cache_text steps.

    python -m fizgig.scripts.minimax_refmod --dit <h3> --dataset_config <toml>
        --output_dir out --output_name subject --grid 16 --steps 200
        [--vae <video vae> --text_encoder <qwen3-vl> --sample_prompts prompts.txt]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.minimax.refmod import run_refmod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def setup_parser():
    p = argparse.ArgumentParser(description="MiniMax H3 RefMod: encode + optimise a reference mod")
    p.add_argument("--dit", required=True, help="MiniMax H3 DiT (pruned int8 or bf16)")
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", required=True)
    p.add_argument("--grid", default="full",
                   help="'full' (default) keeps every reference on the first reference's latent "
                        "canvas — the mode that carries a face; 8 / 16 / 32 average-pool to that "
                        "many latent cells on the long edge (small, stackable, concept-level).")
    p.add_argument("--sigma_min", type=float, default=0.2,
                   help="optimise only at noise levels in [sigma_min, sigma_max] (default 0.2-0.8, "
                        "measured; -1 for both = H3's own shift-12 density)")
    p.add_argument("--sigma_max", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=200,
                   help="optimisation steps against the frozen base (0 = encode only, the "
                        "node extractor's own result)")
    p.add_argument("--lr", type=float, default=1e-3, help="latent-space AdamW rate (default 1e-3, measured)")
    p.add_argument("--pull", type=float, default=2.0,
                   help="weight of the L2 pull toward the initial encode (default 2.0, measured)")
    p.add_argument("--max_refs", type=int, default=8, help="references stacked into the mod (default 8)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_quant", default="auto", help="auto / int8 / nf4 / hqq")
    p.add_argument("--blocks_to_swap", default="auto")
    p.add_argument("--vae", default=None, help="H3 video VAE (previews decode with it)")
    p.add_argument("--text_encoder", default=None, help="Qwen3-VL-32B (preview prompts)")
    p.add_argument("--sample_prompts", default=None, help="one prompt per line")
    p.add_argument("--sample_width", type=int, default=768)
    p.add_argument("--sample_height", type=int, default=768)
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--preview_every", type=int, default=0,
                   help="also preview every N steps (0 = raw mod + finished mod only)")
    p.add_argument("--turbo_lora_path", default=None, help="Turbo LoRA for few-step previews")
    p.add_argument("--turbo_lora_strength", type=float, default=1.0)
    p.add_argument("--description", default="", help="stored in the mod (the loaders can emit it)")
    p.add_argument("--init_from", default=None,
                   help="start from an existing mod file instead of the caches (re-preview it "
                        "with --steps 0, or keep optimising it)")
    return p


def _read_prompts(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        out = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return out or None


def main():
    from fizgig.utils.device import apply_sim_vram_cap
    apply_sim_vram_cap()
    a = setup_parser().parse_args()
    grid = None if str(a.grid).strip().lower() in ("full", "0", "") else int(str(a.grid).split("x")[0])
    run_refmod(dataset_config=a.dataset_config, output_dir=a.output_dir, output_name=a.output_name,
               dit_path=a.dit, grid=grid, steps=a.steps, lr=a.lr, pull=a.pull, max_refs=a.max_refs,
               seed=a.seed, base_quant=a.base_quant, blocks_to_swap=a.blocks_to_swap,
               vae_path=a.vae, te_path=a.text_encoder, sample_prompts=_read_prompts(a.sample_prompts),
               sample_width=a.sample_width, sample_height=a.sample_height, sample_steps=a.sample_steps,
               sample_seed=a.sample_seed, preview_every=a.preview_every,
               turbo_lora_path=a.turbo_lora_path, turbo_lora_strength=a.turbo_lora_strength,
               description=a.description, init_from=a.init_from,
               sigma_range=((a.sigma_min, a.sigma_max) if a.sigma_min >= 0 and a.sigma_max > 0 else None))


if __name__ == "__main__":
    main()
