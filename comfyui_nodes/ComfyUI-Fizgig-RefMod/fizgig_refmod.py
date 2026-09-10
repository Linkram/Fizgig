"""Fizgig H3 RefMod — one node that loads BOTH halves of a Fizgig-made MiniMax H3 RefMod.

A Fizgig RefMod file is a standard RefMod (the ComfyUI-MiniMaxH3Mod format: one `latent` tensor
plus a `refmod_meta` header) that may also carry a rank-2 companion LoRA in the same
`.safetensors` (kohya `lora_unet_*` keys). The standard Load H3 RefMods -> Apply H3 RefMod chain
reads the mod and ignores the LoRA. This node reads both: the mod goes into the conditioning
as a native H3 reference block (the `minimax_refs` list the core Reference-to-Video node and
the mod pack both write), and the LoRA is patched onto the model through ComfyUI's own LoRA
loader. Use it with the Reference (ref2va) checkpoint, the model the pair was fitted against.
"""
import glob
import json
import os

import torch
import torch.nn.functional as F

import comfy.sd
import comfy.utils
import folder_paths

META_KEY = "refmod_meta"
LORA_PREFIX = "lora_unet_"


def refmods_dir() -> str:
    d = os.path.join(folder_paths.models_dir, "refmods")
    os.makedirs(d, exist_ok=True)
    return d


try:
    folder_paths.add_model_folder_path("refmods", refmods_dir())
except Exception:
    pass


def _list_mods():
    names = []
    for p in sorted(glob.glob(os.path.join(glob.escape(refmods_dir()), "*.safetensors"))):
        names.append(os.path.splitext(os.path.basename(p))[0])
    return names or ["(no mods in models/refmods)"]


def _blur_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Heavy spatial low-pass (down then up) — the weakening target for a strength below 1.0,
    the same rule the mod pack uses, so a strength means the same thing in both nodes: a blurred
    copy stays on the latent manifold while shedding the detail that makes a reference strong."""
    if z.dim() != 5:
        return z
    t, h, w = z.shape[2], z.shape[3], z.shape[4]
    sh, sw = max(1, h // factor), max(1, w // factor)
    down = F.adaptive_avg_pool3d(z.float(), (t, sh, sw))
    up = F.interpolate(down, size=(t, h, w), mode="trilinear", align_corners=False)
    return up.to(z.dtype)


def _ref_block(z: torch.Tensor) -> dict:
    """The native H3 reference block for a mod latent [1, 24, T, H, W]: an image block for one
    frame, a video-kind block for a stacked multi-reference mod (comfy_extras/nodes_minimax_h3.py
    emits exactly these shapes)."""
    T = int(z.shape[2])
    if T == 1:
        return {"kind": "image", "latent_h": int(z.shape[3]), "latent_w": int(z.shape[4]), "latent": z}
    return {"kind": "video", "latent_t": T, "latent_h": int(z.shape[3]), "latent_w": int(z.shape[4]),
            "latent": z, "ref_audio_t": 0, "audio_latent": None}


class FizgigH3RefMod:
    """MODEL + CONDITIONING in, MODEL + CONDITIONING out. Reference strength 0-1 (0 = mod not
    injected), LoRA strength 0-2 (0 = LoRA not applied). Chains with the mod pack's Apply and
    Step Curve nodes, which accept native conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "refmod": (_list_mods(),),
                "ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "conditioning", "info")
    FUNCTION = "apply"
    CATEGORY = "Fizgig"
    DESCRIPTION = ("Loads a Fizgig MiniMax H3 RefMod: the reference mod into the conditioning (native "
                   "reference block) and, when the file carries one, the rank-2 companion LoRA onto "
                   "the model. Standard RefMod files work too (mod only). Use the ref2va checkpoint.")

    @classmethod
    def IS_CHANGED(cls, refmod, **kwargs):
        p = os.path.join(refmods_dir(), f"{refmod}.safetensors")
        try:
            st = os.stat(p)
            return f"{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            return ""

    def apply(self, model, conditioning, refmod, ref_strength, lora_strength):
        path = os.path.join(refmods_dir(), f"{refmod}.safetensors")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"RefMod not found: {path}")
        sd, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        if "latent" not in sd:
            raise ValueError(f"{os.path.basename(path)} has no `latent` tensor — not a RefMod")
        latent = sd.pop("latent").float()
        if latent.dim() == 4:
            latent = latent.unsqueeze(2)
        lora = {k: v for k, v in sd.items() if k.startswith(LORA_PREFIX)}
        try:
            rm = json.loads((meta or {}).get(META_KEY, "{}"))
        except Exception:
            rm = {}

        out_cond = conditioning
        if ref_strength > 0.0:
            z = latent
            if ref_strength < 1.0:
                z = ref_strength * z + (1.0 - ref_strength) * _blur_latent(z)
            blk = _ref_block(z)
            out_cond = []
            for t in conditioning:
                d = dict(t[1])
                d["minimax_refs"] = list(d.get("minimax_refs", [])) + [blk]
                out_cond.append([t[0], d])

        out_model = model
        if lora and lora_strength != 0.0:
            out_model, _ = comfy.sd.load_lora_for_models(model, None, lora, lora_strength, 0,
                                                         lora_metadata=meta)

        T = int(latent.shape[2])
        tokens = T * (int(latent.shape[3]) // 2) * (int(latent.shape[4]) // 2)
        info = (f"{refmod}: {rm.get('kind', 'image')} mod, {T} frame(s), {tokens} tokens, "
                f"ref {ref_strength:.2f}"
                + (f"; companion LoRA rank {(meta or {}).get('ss_network_dim', '?')} "
                   f"({len(lora)} tensors) at {lora_strength:.2f}" if lora else "; no companion LoRA in file")
                + (f"; optimised {rm.get('optimize_steps', 0)} steps" if rm.get("optimize_steps") else ""))
        print(f"[Fizgig H3 RefMod] {info}")
        return (out_model, out_cond, info)


NODE_CLASS_MAPPINGS = {"FizgigH3RefMod": FizgigH3RefMod}
NODE_DISPLAY_NAME_MAPPINGS = {"FizgigH3RefMod": "Fizgig H3 RefMod (mod + companion LoRA)"}
