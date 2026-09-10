"""MiniMax H3 RefMods — make one, and make it stronger than a plain encode.

A RefMod (the community node pack `ComfyUI-MiniMaxH3Mod`) is a reference for H3 saved as a
file: the reference run through the video VAE once, its normalized latent stored in a
`.safetensors` with a JSON header, and fed back at generation time as reference condition rows
(H3's own r2v `refs` payload). Nothing in it is learned — the model copies what the rows show.

This module writes that exact file, and then does the thing the extractor can't: with the
H3 base loaded and FROZEN, the mod's latent is the only parameter, and a few hundred steps of
the ordinary flow-matching loss on the user's dataset push it toward the values that make H3
reproduce the subject across the dataset's captions. Textual inversion, in the reference
channel. The output is still just a latent tensor in the node's own format, so the official
loader uses it unchanged — it simply carries more of the subject than the VAE alone did.

Layout contract (checked against the node pack's core.py, 9 Sep 2026, `_format_version` 2):
  tensor  `latent`  [1, 24, T, H, W] fp16, normalized VAE units (ours are the same units —
                    the caches here are std ~1.0, and so is the node's example mod)
  header  `refmod_meta` = JSON {name, kind (image|video), latent_h, latent_w, latent_t, mode
                    (encode|training), source, source_shape, pool, optimize_steps, tags,
                    description, concept_type, _format_version}
A multi-reference mod is ONE video-kind block of T frames (kind "video", latent_t T) — the
node stacks references along time and ComfyUI packs the block on the video clock. The
optimiser feeds the DiT the same single block, so what it learns is what the node will show.
"""
from __future__ import annotations

import gc
import glob
import json
import logging
import math
import os
import random
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

NODE_META_KEY = "refmod_meta"
NODE_FORMAT_VERSION = 2
NODE_TOKEN_CAP = 5120        # the node pack's Extract default `max_tokens`; its loader does not cap
ARCH = "minimaxh3"
MAX_REFS_DEFAULT = 8


# ─── references from the caches ─────────────────────────────────────────────────────────────

def collect_refs(cache_dirs, max_refs: int = MAX_REFS_DEFAULT) -> List[Tuple[str, torch.Tensor, str]]:
    """The dataset's own H3 latent caches -> up to `max_refs` (name, latent [24, h, w], kind).

    Photos come first (each a normalized still latent at its bucket size), then clip stills
    (the sharpest-face frame minimax_cache_latents picked and encoded as `still_latent`).
    A clip cached without a still contributes nothing — a whole clip is not a reference frame.
    Sorted by name so the pick is stable run to run."""
    from safetensors.torch import load_file
    photos, stills = [], []
    seen = set()
    for d in cache_dirs:
        if not d or not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(glob.escape(d), f"*_{ARCH}.safetensors"))):
            base = os.path.basename(p)
            if base in seen:
                continue
            seen.add(base)
            try:
                sd = load_file(p)
            except Exception as exc:
                logger.warning(f"[refmod] skipped {base}: {exc}")
                continue
            stem = base[: -len(f"_{ARCH}.safetensors")]
            if "still_latent" in sd:
                z = sd["still_latent"]
                if z.dim() == 3:
                    stills.append((stem, z.float(), "clip still"))
                continue
            lat_keys = [k for k in sd if k.startswith("latent_")]
            if not lat_keys:
                continue
            z = sd[lat_keys[0]]
            if z.dim() == 3:                       # (C, H, W): a still; clips are 4-D
                photos.append((stem, z.float(), "photo"))
    refs = (photos + stills)[:max_refs]
    return refs


def exclude_refs_from_training(group, ref_stems) -> Tuple[int, int]:
    """Hold the reference stills OUT of the training set: the mod optimiser and the companion
    LoRA must not train on the very photos the mod shows the model (Peter, 10 Sep 2026 — the
    LoRA would just learn to copy what the reference block already carries). Rebuilds each
    dataset's bucket manager without those items. A clip shares its still's cache file, so a
    clip whose still is a reference goes too (clips are skipped by the optimisers anyway).
    Returns (items removed, items remaining)."""
    from fizgig.dataset.image_dataset import BucketBatchManager
    stems = set(ref_stems)
    suffix = f"_{ARCH}.safetensors"
    removed = 0
    for ds in group.datasets:
        bm = getattr(ds, "batch_manager", None)
        if bm is None:
            continue
        kept = {}
        for reso, items in bm.buckets.items():
            keep = []
            for it in items:
                base = os.path.basename(getattr(it, "latent_cache_path", "") or "")
                stem = base[: -len(suffix)] if base.endswith(suffix) else base
                if stem in stems:
                    removed += 1
                else:
                    keep.append(it)
            if keep:
                kept[reso] = keep
        ds.batch_manager = BucketBatchManager(kept, bm.batch_size, num_timestep_buckets=bm.num_timestep_buckets)
        ds.num_train_items = sum(len(b) for b in kept.values())
    group.num_train_items = sum(getattr(ds, "num_train_items", 0) for ds in group.datasets)
    # The group is a torch ConcatDataset: its index table was built from the OLD batch counts
    # (the first held-out run indexed past the end). Recompute it.
    try:
        from torch.utils.data import ConcatDataset
        if isinstance(group, ConcatDataset):
            group.cumulative_sizes = ConcatDataset.cumsum(group.datasets)
    except Exception:
        pass
    return removed, group.num_train_items


def attach_mod_to_file(path: str, latent: torch.Tensor, *, name: str, pool: str, source_shape: str = "",
                       tags=None, description: str = "", extra: Optional[dict] = None) -> str:
    """Rewrite a saved LoRA `.safetensors` (kohya `lora_unet_*` keys + metadata) as a RefMod PAIR
    file: the same tensors plus the mod's `latent` and the node pack's `refmod_meta` header.
    Used by the H3 trainer's RefMod mode on every checkpoint and the final save, so each file
    the run writes is a standard RefMod to their loader and a mod+LoRA pair to ours."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    tensors, header = {}, {}
    with safe_open(path, framework="pt", device="cpu") as f:
        header = dict(f.metadata() or {})
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    lat = latent.detach().to("cpu", torch.float16).contiguous()
    if lat.dim() == 4:
        lat = lat.unsqueeze(2)
    T = int(lat.shape[2])
    meta = {
        "name": name, "kind": "video" if T > 1 else "image",
        "latent_h": int(lat.shape[3]), "latent_w": int(lat.shape[4]), "latent_t": T,
        "mode": "encode" if "full-res" in str(pool) else "training",
        "source": "stack" if T > 1 else "image", "source_shape": source_shape, "pool": pool,
        "optimize_steps": 0, "tags": list(tags or []), "description": description or "",
        "concept_type": "identity", "_format_version": NODE_FORMAT_VERSION,
    }
    header[NODE_META_KEY] = json.dumps(meta)
    for k, v in (extra or {}).items():
        header[str(k)] = str(v)
    tensors["latent"] = lat
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata=header)
    os.replace(tmp, path)
    return path


def aspect_grid(pool: int, aspect_hw: float) -> Tuple[int, int]:
    """Even (h, w) latent grid whose long edge is `pool` and whose aspect matches the source —
    the node's own rule (a portrait pooled into a square grid comes out "fat")."""
    if aspect_hw >= 1.0:
        h, w = float(pool), pool / aspect_hw
    else:
        w, h = float(pool), pool * aspect_hw
    return max(2, round(h / 2) * 2), max(2, round(w / 2) * 2)


def build_mod(refs, grid: Optional[int]) -> Tuple[torch.Tensor, str]:
    """(name, [24, h, w], kind) refs -> the mod latent [1, 24, T, gh, gw] fp32 + a pool label.

    grid = None keeps each reference at full resolution on the FIRST reference's latent canvas
    (the node's encode mode cover-crops every ref to one canvas; here they are resampled to
    it, which for latents is the same operation the node's own refinement uses). An integer
    grid average-pools every reference to that many latent cells on its long edge, aspect
    kept from the first reference (the node's training mode). Dims are always even: the DiT
    patches 2x2 latent cells into one token."""
    if not refs:
        raise ValueError("no references — cache the dataset first (photos, or clips with "
                         "the sharpest-face still on)")
    h0, w0 = int(refs[0][1].shape[-2]), int(refs[0][1].shape[-1])
    if grid is None:
        gh, gw = (h0 // 2) * 2, (w0 // 2) * 2
        label = f"full-res {gw * 16}x{gh * 16}px"
    else:
        gh, gw = aspect_grid(int(grid), h0 / float(w0))
        label = f"{len(refs)}x{gh}x{gw}"
    frames = []
    for _, z, _ in refs:
        z4 = z.float().unsqueeze(0)                                   # [1, 24, h, w]
        if grid is None:
            if (z4.shape[-2], z4.shape[-1]) != (gh, gw):
                z4 = F.interpolate(z4, size=(gh, gw), mode="bilinear", align_corners=False)
        else:
            z4 = F.adaptive_avg_pool2d(z4, (gh, gw))
        frames.append(z4)
    latent = torch.stack(frames, dim=2)                               # [1, 24, T, gh, gw]
    return latent.contiguous(), label


# ─── the file ────────────────────────────────────────────────────────────────────────────────

def save_refmod(path_no_ext: str, latent: torch.Tensor, *, name: str, mode: str, pool: str,
                optimize_steps: int, source_shape: str = "", tags=None, description: str = "",
                concept_type: str = "identity", extra: Optional[dict] = None,
                lora_sd: Optional[dict] = None) -> str:
    """Write `<path>.safetensors` in the node pack's layout (+ Fizgig's own `ss_*` keys, which
    the node ignores). `lora_sd` (kohya `lora_unet_*` tensors, the companion LoRA) rides in the
    SAME file: their loader reads only `latent` + the `refmod_meta` header, so the file stays a
    standard RefMod for them and a mod+LoRA pair for the Fizgig node. Returns the path."""
    from safetensors.torch import save_file
    latent = latent.detach().to("cpu", torch.float16).contiguous()
    if latent.dim() == 4:
        latent = latent.unsqueeze(2)
    assert latent.dim() == 5 and latent.shape[0] == 1, f"latent must be [1, 24, T, H, W], got {tuple(latent.shape)}"
    T = int(latent.shape[2])
    meta = {
        "name": name,
        "kind": "video" if T > 1 else "image",
        "latent_h": int(latent.shape[3]),
        "latent_w": int(latent.shape[4]),
        "latent_t": T,
        "mode": mode,
        "source": "stack" if T > 1 else "image",
        "source_shape": source_shape,
        "pool": pool,
        "optimize_steps": int(optimize_steps),
        "tags": list(tags or []),
        "description": description or "",
        "concept_type": concept_type,
        "_format_version": NODE_FORMAT_VERSION,
    }
    header = {NODE_META_KEY: json.dumps(meta)}
    for k, v in (extra or {}).items():
        header[str(k)] = str(v)
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    out = path_no_ext + ".safetensors"
    tensors = {"latent": latent}
    if lora_sd:
        for k, v in lora_sd.items():
            if k == "latent" or not k.startswith("lora_unet_"):
                raise ValueError(f"companion LoRA key {k!r} is not a lora_unet_* tensor")
            tensors[k] = v.detach().to("cpu").contiguous()
    save_file(tensors, out, metadata=header)
    return out


def load_refmod(path: str):
    """-> (latent [1, 24, T, H, W] fp32, meta dict, lora_sd or None) — the node's reader, in
    miniature, plus the Fizgig half."""
    from safetensors import safe_open
    lora_sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = json.loads((f.metadata() or {}).get(NODE_META_KEY, "{}"))
        latent = f.get_tensor("latent").float().clone()
        for k in f.keys():
            if k.startswith("lora_unet_"):
                lora_sd[k] = f.get_tensor(k).clone()
    return latent, meta, (lora_sd or None)


def token_count(latent: torch.Tensor) -> int:
    return int(latent.shape[2]) * (int(latent.shape[3]) // 2) * (int(latent.shape[4]) // 2)


# ─── the base model, planned like a training run ────────────────────────────────────────────

def plan_and_load_dit(dit_path: str, *, device, dtype, base_quant: str = "auto",
                      blocks_to_swap="auto", mp: float = 0.25):
    """The H3 base on the tier a LoRA run would get (int8 no-swap / int8 streamed / NF4),
    frozen, gradient-checkpointed. No adapter: the plan's adapter budget is zero."""
    from fizgig.minimax.loader import load_minimax_h3_dit
    from fizgig.minimax.trainer import (is_pruned_checkpoint, plan_base_quant, plan_vram,
                                        _INT8_TRANSIENT_GB)
    from fizgig.minimax import trainer as _tr
    pruned = is_pruned_checkpoint(dit_path)
    mode, n_swap = base_quant, 0
    if str(blocks_to_swap).lower() == "auto":
        if torch.cuda.is_available():
            from fizgig.utils.device import plannable_free_vram
            free_gb = plannable_free_vram()
            if base_quant == "auto":
                mode, n_swap, _ckpt, why = plan_base_quant(free_gb, pruned, mp=mp, adapter_gb=0.0)
            else:
                mode = base_quant
                resident = (_tr._RESIDENT_INT8_GB if mode == "int8"
                            else (_tr._RESIDENT_PRUNED_GB if pruned else _tr._RESIDENT_GB))
                n_swap, _ckpt = plan_vram(free_gb, mp=mp, resident_gb=resident,
                                          transient_gb=_INT8_TRANSIENT_GB if mode == "int8" else 0.0,
                                          adapter_gb=0.0)
                why = f"base precision pinned to {mode}"
            logger.info(f"[vram] refmod plan: free {free_gb:.1f} GB, largest bucket {mp:.2f} MP, "
                        f"base {mode} -> blocks_to_swap={n_swap} ({why})")
        else:
            mode = "nf4" if base_quant == "auto" else base_quant
    else:
        n_swap = int(blocks_to_swap)
        mode = ("int8" if pruned else "nf4") if base_quant == "auto" else base_quant
    if not pruned and mode == "int8":
        mode = "nf4"
    dit = load_minimax_h3_dit(dit_path, device=device, compute_dtype=dtype, quantize=True,
                              blocks_to_swap=n_swap, base_quant=mode, adaln_fp32=True)
    dit.requires_grad_(False)
    if n_swap > 0:
        h2d = mode == "int8" or (mode in ("nf4", "hqq") and os.environ.get("FIZGIG_NO_NF4_H2D") != "1")
        n_swap = dit.enable_block_swap(n_swap, h2d_only=h2d, ring_size=2)
    dit.enable_gradient_checkpointing()
    dit.eval()
    return dit, mode, n_swap


# ─── the optimisation ────────────────────────────────────────────────────────────────────────

def refmod_step_loss(dit, mod: torch.Tensor, latents: torch.Tensor, text: torch.Tensor, *,
                     device, dtype, shift=None, generator=None, seed: int = 0,
                     sigma_range=None):
    """One flow-matching loss with the mod riding as the reference block. `mod` is the
    parameter ([1, 24, T, gh, gw] fp32, requires_grad); grads reach it through the DiT's
    condition rows (the frozen base only supplies dX)."""
    from fizgig.minimax.trainer import sample_sigmas
    x0 = latents.float()
    _pt, _ph, _pw = getattr(dit, "patch_size", (1, 2, 2))
    H, W = x0.shape[-2], x0.shape[-1]
    Hc, Wc = (H // _ph) * _ph, (W // _pw) * _pw
    if (Hc, Wc) != (H, W):
        x0 = x0[..., :Hc, :Wc].contiguous()
    noise = torch.randn(x0.shape, device=device, generator=generator, dtype=torch.float32)
    tokens = (x0.shape[-2] // _ph) * (x0.shape[-1] // _pw)
    if sigma_range:
        lo, hi = float(sigma_range[0]), float(sigma_range[1])
        sigma = lo + (hi - lo) * torch.rand(1, device=device, generator=generator)
    else:
        sigma = sample_sigmas(1, device, shift=shift, generator=generator, image_tokens=tokens)
    s = sigma.reshape(1, 1, 1, 1, 1)
    noised = (1.0 - s) * x0 + s * noise
    t = (1.0 - sigma).to(device)
    pred = dit(noised.to(dtype), t, text, ref_latents=[mod], seed=seed)
    return F.mse_loss(pred.float(), (x0 - noise).float()), float(sigma.reshape(-1)[0])


DEFAULT_LR = 1e-3
DEFAULT_PULL = 2.0
DEFAULT_SIGMA_RANGE = (0.2, 0.8)


def optimize_refmod(dit, group, mod0: torch.Tensor, *, steps: int, lr: float = DEFAULT_LR,
                    pull: float = DEFAULT_PULL, device="cuda", dtype=torch.bfloat16, seed: int = 42,
                    uncond_text: Optional[torch.Tensor] = None, uncond_frac: float = 0.1,
                    warmup: int = 20, log_every: int = 10, on_step=None,
                    target: Optional[torch.Tensor] = None, shared_epoch=None,
                    sigma_range=DEFAULT_SIGMA_RANGE) -> torch.Tensor:
    """Optimise the mod latent against the frozen H3 loss over the dataset's stills.

    Defaults are the measured recipe (mbacc photos, 10 Sep 2026, ref2va, Full canvas, 4 seeds,
    ArcFace vs the dataset): lr 1e-3, pull 2.0, noise window 0.2-0.8 put every seed at or
    above the raw encode on portrait prompts (70 vs 66) and +5.5 on four off-dataset scene
    prompts (58.7 vs 53.2). lr 5e-3 / pull 0.5 on H3's full shift-12 density LOST fidelity
    (55): at the top of that schedule the loss is about global structure, and pushing the
    reference rows to serve it costs the face detail they exist to carry.

    pull is the weight of an L2 term toward the initial encode: it keeps the mod a reference
    (on the VAE manifold, so the node's blur-toward-itself strength control still means what
    it means) rather than letting it walk off into an adversarial pattern that only works with
    the training captions. uncond_frac of the steps use the empty-prompt embedding when the
    cache has one, so the mod learns to carry the subject without a particular caption."""
    from torch.utils.data import DataLoader
    from fizgig.minimax.trainer import _Collator
    from multiprocessing import Value
    if steps <= 0:
        return mod0.detach().clone()
    torch.manual_seed(seed)
    random.seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    # the SAME counter the dataset group was built with — its buckets assert on it
    if shared_epoch is None:
        shared_epoch = getattr(group, "_fizgig_shared_epoch", None) or Value("i", 0)
    loader = DataLoader(group, batch_size=1, shuffle=True, collate_fn=_Collator(shared_epoch, group),
                        num_workers=0)
    target = (target if target is not None else mod0).detach().to(device, torch.float32)
    param = torch.nn.Parameter(mod0.detach().to(device, torch.float32).clone())
    opt = torch.optim.AdamW([param], lr=lr, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8)
    step = 0
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    skipped_clips = 0
    while step < steps:
        shared_epoch.value += 1
        for batch in loader:
            if step >= steps:
                break
            lat = batch["latents"]
            if lat.dim() != 4:                     # a whole clip (or a voice placeholder)
                skipped_clips += 1
                continue
            latents = lat.to(device, dtype).unsqueeze(2)              # (1, 24, 1, H, W)
            text = batch["hidden_states"].to(device, dtype)
            if uncond_text is not None and random.random() < uncond_frac:
                text = uncond_text.to(device, dtype)
            # linear warm-up, then flat — the mod starts ON the data, a full stride at step 0
            # is the one thing that reliably breaks it
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / float(max(1, warmup)))
            with torch.autocast("cuda", enabled=False):
                loss, sig = refmod_step_loss(dit, param, latents, text, device=device, dtype=dtype,
                                             generator=gen, seed=seed, sigma_range=sigma_range)
                total = loss + pull * F.mse_loss(param, target) if pull > 0 else loss
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([param], 1.0)
            opt.step()
            step += 1
            run_loss += float(loss.detach())
            run_n += 1
            if on_step is not None:
                on_step(step, float(loss.detach()))
            if log_every and (step % log_every == 0 or step == steps):
                drift = float((param.detach() - target).pow(2).mean().sqrt())
                el = time.time() - t0
                print(f"[refmod] step {step}/{steps}  loss {run_loss / max(1, run_n):.4f}  "
                      f"drift {drift:.3f} (rms, latent units)  {el / step:.2f} s/step", flush=True)
                run_loss, run_n = 0.0, 0
    if skipped_clips:
        logger.info(f"[refmod] {skipped_clips} clip item(s) skipped — the optimiser trains on "
                    f"stills (photos and clip stills)")
    return param.detach().to("cpu", torch.float32)


# ─── the companion LoRA ─────────────────────────────────────────────────────────────────────

COMPANION_DIM = 2
COMPANION_ALPHA = 2
COMPANION_LR = 2e-4


def train_companion_lora(dit, group, mod: torch.Tensor, *, epochs: int = 2, lr: float = COMPANION_LR,
                         dim: int = COMPANION_DIM, alpha: float = COMPANION_ALPHA, device="cuda",
                         dtype=torch.bfloat16, seed: int = 42, uncond_text=None, uncond_frac: float = 0.05,
                         shared_epoch=None, log_every: int = 10):
    """A rank-`dim` LoRA over the H3 blocks, trained for `epochs` passes over the dataset stills
    WITH the mod riding as the reference block. The mod is fixed, so the LoRA learns only what
    the reference channel cannot express — the subject away from the reference's own pose,
    expression and scene. Returns the network, its modules switched OFF (multiplier 0) so the
    caller decides when it is in the picture; `network.state_dict()` is the kohya-keyed half
    that goes into the mod file. H3's own noise density (no window): the LoRA is meant to
    carry the subject at every step, the way a trained LoRA does."""
    from torch.utils.data import DataLoader
    from multiprocessing import Value
    from fizgig.networks.lora import create_network
    from fizgig.minimax.trainer import _Collator, DEFAULT_INCLUDE_PATTERNS
    net = create_network(None, "lora_unet", 1.0, int(dim), float(alpha), None, [], dit,
                         include_patterns=list(DEFAULT_INCLUDE_PATTERNS))
    net.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    net.requires_grad_(True)
    net.to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in net.parameters())
    logger.info(f"[refmod] companion LoRA: rank {dim}, {len(net.unet_loras)} Linears, "
                f"{n_params / 1e6:.1f} M params, {epochs} epoch(s) at lr {lr:g}")
    if epochs <= 0:
        for m in net.unet_loras:
            m.multiplier = 0.0
        return net
    torch.manual_seed(seed)
    random.seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    if shared_epoch is None:
        shared_epoch = getattr(group, "_fizgig_shared_epoch", None) or Value("i", 0)
    loader = DataLoader(group, batch_size=1, shuffle=True, collate_fn=_Collator(shared_epoch, group),
                        num_workers=0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    ref = mod.detach().to(device, torch.float32)
    step = 0
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    for ep in range(int(epochs)):
        shared_epoch.value += 1
        for batch in loader:
            lat = batch["latents"]
            if lat.dim() != 4:
                continue
            latents = lat.to(device, dtype).unsqueeze(2)
            text = batch["hidden_states"].to(device, dtype)
            if uncond_text is not None and random.random() < uncond_frac:
                text = uncond_text.to(device, dtype)
            with torch.autocast("cuda", enabled=False):
                loss, _ = refmod_step_loss(dit, ref, latents, text, device=device, dtype=dtype,
                                           generator=gen, seed=seed, sigma_range=None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            run_loss += float(loss.detach())
            run_n += 1
            if log_every and step % log_every == 0:
                print(f"[refmod-lora] epoch {ep + 1}/{epochs} step {step}  loss {run_loss / max(1, run_n):.4f}  "
                      f"{(time.time() - t0) / step:.2f} s/step", flush=True)
                run_loss, run_n = 0.0, 0
    for m in net.unet_loras:
        m.multiplier = 0.0
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    return net


def companion_state_dict(net, dtype=torch.bfloat16) -> dict:
    return {k: v.detach().to("cpu", dtype).contiguous() for k, v in net.state_dict().items()}


# ─── previews ────────────────────────────────────────────────────────────────────────────────

def render_previews(dit, mod: torch.Tensor, encoded_prompts, *, out_dir: str, output_name: str,
                    epoch: int, width: int, height: int, steps: int, seed: int, device, dtype,
                    decoder=None, n_swap: int = 0, turbo=None, lora_net=None, lora_on: bool = False):
    """One still per prompt with the mod as the reference block — the way the node will use
    it (no <Picture> vision blocks: the file has no vision side). PNG names follow the
    gallery's contract `<name>_e<epoch>_<i>_<ts>_<seed>.png`."""
    from PIL import Image
    from fizgig.minimax import sampling
    from fizgig.minimax.trainer import (park_dit_to_cpu, restore_parked_dit, turbo_adaln_patch,
                                        turbo_adaln_unpatch, cap_preview_res_small_card)
    os.makedirs(out_dir, exist_ok=True)
    width, height = cap_preview_res_small_card(width, height)
    ts = time.strftime("%Y%m%d%H%M%S")
    ref = mod.to(device, dtype)
    turbo_net, turbo_adaln = (turbo if turbo else (None, []))
    rendered = []
    if lora_net is not None:
        for m in lora_net.unet_loras:
            m.multiplier = 1.0 if lora_on else 0.0
    try:
        if turbo_net is not None:
            turbo_net.to(device=device, dtype=dtype)
            for m in turbo_net.unet_loras:
                m.enabled = True
            turbo_adaln_patch(dit, turbo_adaln, device, dtype)
        with torch.no_grad():
            for i, txt in enumerate(encoded_prompts):
                print(f"[preview] refmod preview {epoch}: prompt {i + 1}/{len(encoded_prompts)} "
                      f"({width}x{height}, seed {seed + i})", flush=True)
                lat, _ = sampling.sample_image(dit, txt.to(device, dtype), width=width, height=height,
                                               steps=steps, cfg_scale=1.0, seed=seed + i,
                                               device=device, dtype=dtype, log_steps=False,
                                               num_frames=1, ref_latents=[ref], return_audio=True)
                rendered.append((f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}", lat.to("cpu")))
                del lat
    finally:
        if turbo_net is not None:
            for m in turbo_net.unet_loras:
                m.enabled = False
            turbo_adaln_unpatch(turbo_adaln)
            turbo_net.to("cpu")
        if lora_net is not None:
            for m in lora_net.unet_loras:
                m.multiplier = 0.0
    del ref
    gc.collect()
    torch.cuda.empty_cache()
    # decode: on 16 GB-class cards the whole base parks for the VAE (same rule as training)
    parked = False
    try:
        small = torch.cuda.get_device_properties(0).total_memory / 1e9 < 20.0
    except Exception:
        small = False
    if decoder is not None and small:
        park_dit_to_cpu(dit)
        parked = True
        gc.collect()
        torch.cuda.empty_cache()
    try:
        if decoder is not None:
            decoder = decoder.to(device)
        with torch.no_grad():
            for stem, lat in rendered:
                if decoder is not None:
                    px = decoder.decode(lat.to(device).float())[0]
                    arr = (px.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    img = Image.fromarray(arr)
                else:
                    arr = sampling.latent_to_rgb(lat)
                    img = Image.fromarray(arr).resize((width, height), Image.NEAREST)
                img.save(os.path.join(out_dir, stem + ".png"))
    finally:
        if decoder is not None:
            decoder.to("cpu")
        if parked:
            restore_parked_dit(dit, device, n_swap)
        gc.collect()
        torch.cuda.empty_cache()
    return [os.path.join(out_dir, s + ".png") for s, _ in rendered]


# ─── the run ─────────────────────────────────────────────────────────────────────────────────

def run_refmod(*, dataset_config: str, output_dir: str, output_name: str, dit_path: str,
               grid: Optional[int] = None, steps: int = 200, lr: float = DEFAULT_LR, pull: float = DEFAULT_PULL,
               max_refs: int = MAX_REFS_DEFAULT, seed: int = 42, base_quant: str = "auto",
               blocks_to_swap="auto", vae_path: Optional[str] = None,
               te_path: Optional[str] = None, sample_prompts: Optional[List[str]] = None,
               sample_width: int = 768, sample_height: int = 768, sample_steps: int = 20,
               sample_seed: int = 42, preview_every: int = 0,
               turbo_lora_path: Optional[str] = None, turbo_lora_strength: float = 1.0,
               description: str = "", init_from: Optional[str] = None,
               sigma_range=DEFAULT_SIGMA_RANGE, companion_lora_epochs: int = 0,
               companion_lora_lr: float = COMPANION_LR, companion_lora_rank: int = COMPANION_DIM,
               exclude_refs: bool = True) -> str:
    """Make the mod, optimise it, write it. Returns the output path.

    One file: <output_dir>/<output_name>.safetensors. Steps = 0 writes the plain encode (the
    node extractor's own result); otherwise the optimised mod. Previews: one set before any
    step (epoch 0) and one from the finished mod, plus every `preview_every` steps between."""
    import argparse
    from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,
                                       generate_dataset_group_by_blueprint, load_user_config)
    from fizgig.dataset.image_dataset import ImageDataset
    from fizgig.training.metadata import ARCHITECTURE_MINIMAX
    from multiprocessing import Value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # dataset (stills; clip stills as photos so every clip lends its sharpest face)
    ImageDataset.clip_still_as_photo = True
    user_config = load_user_config(dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, argparse.Namespace(), architecture=ARCHITECTURE_MINIMAX)
    shared_epoch = Value("i", 0)
    group = generate_dataset_group_by_blueprint(
        blueprint.dataset_group, training=True, num_timestep_buckets=None, shared_epoch=shared_epoch)
    group._fizgig_shared_epoch = shared_epoch
    if group.num_train_items == 0:
        raise RuntimeError("No training items — run the MiniMax cache steps first.")
    cache_dirs = [getattr(ds, "cache_directory", "") for ds in group.datasets]
    refs = collect_refs(cache_dirs, max_refs=max_refs)
    if not refs:
        raise RuntimeError("No reference stills in the caches (photos, or clips cached with "
                           "'Also train the sharpest face still').")
    n_img = sum(1 for r in refs if r[2] == "photo")
    n_st = len(refs) - n_img
    mp = max((r[1].shape[-2] * 16) * (r[1].shape[-1] * 16) for r in refs) / 1e6
    _init_lora = None
    if init_from:
        # start from an existing mod (re-preview it, or keep optimising it)
        mod0, _m0, _init_lora = load_refmod(init_from)
        if _init_lora and companion_lora_epochs <= 0:
            logger.info(f"[refmod] {init_from} carries a companion LoRA ({len(_init_lora)} tensors) — "
                        f"kept in the output as-is; previews here run without it")
        pool_label = str(_m0.get("pool", "")) or f"{mod0.shape[2]}x{mod0.shape[3]}x{mod0.shape[4]}"
        mode = str(_m0.get("mode", "training"))
        source_shape = str(_m0.get("source_shape", ""))
        logger.info(f"[refmod] starting from {init_from}: mod {tuple(mod0.shape)} "
                    f"({token_count(mod0)} tokens, {_m0.get('optimize_steps', 0)} prior steps)")
    else:
        logger.info(f"[refmod] {len(refs)} reference(s): {n_img} photo(s), {n_st} clip still(s) — "
                    + ", ".join(r[0] for r in refs))
        mod0, pool_label = build_mod(refs, grid)
        mode = "training" if grid is not None else "encode"
        _tok = token_count(mod0)
        logger.info(f"[refmod] mod {tuple(mod0.shape)} ({pool_label}, {_tok} tokens, "
                    f"mode {mode}) — the node pack's extractor caps at {NODE_TOKEN_CAP} by default")
        if _tok > NODE_TOKEN_CAP:
            logger.warning(f"[refmod] {_tok} tokens is ABOVE the standard extractor's default cap "
                           f"({NODE_TOKEN_CAP}). The loaders don't refuse it, but every one of "
                           f"those tokens rides in the sequence at each sampling step — slower and "
                           f"more VRAM at generation. Fewer References or a pooled Grid brings it "
                           f"down.")
        source_shape = " +".join(f"1x{r[1].shape[-2]}x{r[1].shape[-1]}" for r in refs)

    if exclude_refs and (steps > 0 or companion_lora_epochs > 0):
        _rm, _left = exclude_refs_from_training(group, [r[0] for r in refs])
        if _left <= 0:
            logger.warning(f"[refmod] every still in the dataset is a reference — nothing would be "
                           f"left to train on, so the references stay in the training set")
            # rebuild is destructive; reload the group as it was
            group = generate_dataset_group_by_blueprint(
                blueprint.dataset_group, training=True, num_timestep_buckets=None, shared_epoch=shared_epoch)
            group._fizgig_shared_epoch = shared_epoch
        else:
            logger.info(f"[refmod] {_rm} reference still(s) held out of training — the optimiser and "
                        f"the companion LoRA train on the other {_left} item(s)")

    uncond_text = None
    for d in cache_dirs:
        f = os.path.join(d or "", f"uncond_{ARCHITECTURE_MINIMAX}_te.safetensors")
        if os.path.isfile(f):
            from safetensors.torch import load_file
            uncond_text = load_file(f)["hidden_states"].unsqueeze(0)
            break

    # prompts BEFORE the DiT (the TE never shares the card with it)
    encoded = None
    if sample_prompts and te_path:
        from fizgig.minimax.sampling import encode_sample_prompts
        logger.info(f"[preview] pre-encoding {len(sample_prompts)} sample prompt(s)...")
        encoded = encode_sample_prompts(te_path, sample_prompts, device=device, quantize=True)

    tags = [f"{n_img} img, {n_st} clip stills", "fizgig"]

    dit, base_mode, n_swap = plan_and_load_dit(dit_path, device=device, dtype=dtype,
                                               base_quant=base_quant, blocks_to_swap=blocks_to_swap, mp=mp)
    decoder = None
    if vae_path and encoded:
        from safetensors import safe_open
        from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder
        decoder = MiniMaxH3VideoVAEDecoder()
        with safe_open(vae_path, framework="pt", device="cpu") as f:
            decoder.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, strict=False)
        decoder = decoder.to(torch.float16).eval()
    turbo = None
    if encoded and turbo_lora_path:
        from fizgig.minimax.trainer import load_preview_turbo
        turbo = load_preview_turbo(dit, turbo_lora_path, turbo_lora_strength)
    sample_dir = os.path.join(output_dir, "sample")

    lora_net = [None]

    def _preview(mod, epoch, lora_on=False):
        if not encoded:
            return
        render_previews(dit, mod, encoded, out_dir=sample_dir, output_name=output_name, epoch=epoch,
                        width=sample_width, height=sample_height, steps=sample_steps, seed=sample_seed,
                        device=device, dtype=dtype, decoder=decoder, n_swap=n_swap, turbo=turbo,
                        lora_net=lora_net[0], lora_on=lora_on)

    _preview(mod0, 0)
    mod = mod0
    if steps > 0:
        logger.info(f"[refmod] optimising {token_count(mod0)} tokens for {steps} steps "
                    f"(lr {lr:g}, pull {pull:g}, base {base_mode}, swap {n_swap})")
        mod = _optimize_with_previews(dit, group, mod0, steps=steps, lr=lr, pull=pull, device=device,
                                      dtype=dtype, seed=seed, uncond_text=uncond_text,
                                      preview_every=preview_every, preview_fn=_preview,
                                      sigma_range=sigma_range)
        _preview(mod, int(math.ceil(steps / float(preview_every))) if preview_every else 1)

    lora_sd = None
    lora_extra = {}
    n_last = (int(math.ceil(steps / float(preview_every))) if preview_every else 1) if steps > 0 else 0
    if companion_lora_epochs > 0:
        # the mod is final now; the LoRA learns the residual around it
        _rank = max(1, int(companion_lora_rank))
        lora_net[0] = train_companion_lora(dit, group, mod, epochs=companion_lora_epochs, lr=companion_lora_lr,
                                           dim=_rank, alpha=_rank,
                                           device=device, dtype=dtype, seed=seed, uncond_text=uncond_text)
        _preview(mod, n_last + 1, lora_on=True)
        lora_sd = companion_state_dict(lora_net[0])
        lora_extra = {"ss_refmod_lora": "1", "ss_network_module": "fizgig.minimax (lora_unet, transformer blocks)",
                      "ss_network_dim": str(_rank), "ss_network_alpha": str(_rank),
                      "ss_refmod_lora_epochs": str(companion_lora_epochs),
                      "ss_refmod_lora_lr": f"{companion_lora_lr:g}", "ss_refmod_lora_strength": "1.0",
                      "ss_architecture": "minimaxh3"}
    elif init_from and _init_lora:
        lora_sd = _init_lora

    out = save_refmod(os.path.join(output_dir, output_name), mod, name=output_name, mode=mode,
                      pool=pool_label, optimize_steps=steps, source_shape=source_shape,
                      tags=tags + (["fizgig optimised"] if steps > 0 else [])
                           + (["fizgig companion lora"] if lora_sd else []),
                      description=description,
                      extra={"ss_refmod_steps": str(steps), "ss_refmod_lr": f"{lr:g}",
                             "ss_refmod_pull": f"{pull:g}", "ss_refmod_refs": str(len(refs)),
                             "ss_refmod_base": base_mode, "ss_refmod_grid": str(grid or "full"),
                             **lora_extra},
                      lora_sd=lora_sd)
    mb = os.path.getsize(out) / 1024 / 1024
    logger.info(f"[refmod] saved {out} ({token_count(mod)} tokens"
                + (f" + companion LoRA rank {(lora_extra.get('ss_network_dim') or '?')}" if lora_sd else "") + f", {mb:.2f} MB) — "
                f"copy it to ComfyUI/models/refmods/: Load H3 RefMods reads the mod"
                + ("; the Fizgig H3 RefMod node loads the mod AND the LoRA" if lora_sd else ""))
    return out


def _optimize_with_previews(dit, group, mod0, *, steps, lr, pull, device, dtype, seed, uncond_text,
                            preview_every, preview_fn, sigma_range=None):
    """optimize_refmod in chunks so interim previews render from the live latent."""
    if not preview_every or preview_every >= steps:
        return optimize_refmod(dit, group, mod0, steps=steps, lr=lr, pull=pull, device=device,
                               dtype=dtype, seed=seed, uncond_text=uncond_text, sigma_range=sigma_range)
    # chunked: each chunk restarts the optimizer state but keeps the latent — a small price,
    # and it keeps optimize_refmod itself simple. Warm-up only on the first chunk.
    mod = mod0
    done = 0
    k = 0
    while done < steps:
        n = min(preview_every, steps - done)
        mod = optimize_refmod(dit, group, mod, steps=n, lr=lr, pull=pull, device=device, dtype=dtype,
                              seed=seed + k, uncond_text=uncond_text, warmup=(20 if k == 0 else 1),
                              target=mod0, sigma_range=sigma_range)
        done += n
        k += 1
        if done < steps:
            preview_fn(mod, k)
    return mod
