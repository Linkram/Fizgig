# ComfyUI-Fizgig-RefMod

One node, **Fizgig H3 RefMod (mod + companion LoRA)**, for RefMod files made by [Fizgig](https://github.com/shootthesound/Fizgig)'s MiniMax H3 RefMod trainer.

A Fizgig RefMod is a standard MiniMax H3 RefMod file (the [ComfyUI-MiniMaxH3Mod](https://github.com/shingo257/comfyui-minimaxh3mod) format: one `latent` tensor plus a `refmod_meta` header) whose latent has been optimised against the frozen H3 model, and which may also carry a small rank-2 **companion LoRA** in the same `.safetensors`, trained with the mod in the conditioning so it holds what a reference can't: the subject away from the reference's own pose, expression and scene.

- The standard **Load H3 RefMods → Apply H3 RefMod** chain reads the mod half of the same file and ignores the LoRA. Nothing about the file is special to them.
- **This node reads both halves.** The mod goes into the conditioning as a native H3 reference block (the same `minimax_refs` list the core Reference-to-Video node writes), and the LoRA is patched onto the model through ComfyUI's own LoRA loader.

## Install

Copy or symlink this folder into `ComfyUI/custom_nodes/` and restart ComfyUI. No dependencies beyond ComfyUI itself. Put mods in `ComfyUI/models/refmods/` (the same folder the mod pack uses).

## Use

`Load Diffusion Model` (the **Reference / ref2va** H3 checkpoint, the model the pair was fitted against) → **Fizgig H3 RefMod** takes `model` and your text `conditioning`, outputs both → sampler as usual.

| input | meaning |
|---|---|
| `refmod` | a file from `models/refmods/` |
| `ref_strength` | 0–1. Below 1 the latent is mixed toward a blurred copy of itself, the mod pack's own rule, so a strength means the same thing in both nodes. 0 leaves the conditioning untouched. |
| `lora_strength` | 0–2 for the companion LoRA. 0 leaves the model untouched. Ignored when the file has no LoRA. |

The `info` output names what was loaded. The mod pack's Apply and Step Curve nodes accept the conditioning this node outputs, so their per-step curves still work on top.

## The file

```
latent                      [1, 24, T, H, W] fp16   — the mod (theirs, unchanged)
lora_unet_*.lora_down.weight / lora_up.weight / alpha — the companion LoRA (kohya keys)
refmod_meta                 JSON header (theirs, unchanged)
ss_*                        Fizgig's training metadata
```
