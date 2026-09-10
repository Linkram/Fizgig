"""MiniMax H3 RefMod (9 Sep 2026): the file matches the community node pack's format, a
multi-frame reference packs on the video clock like ComfyUI's PackedLayout, the mod builder
pools/resamples the way the node does, and the GUI entry is a two-control tab. CPU only.

Run: venv/Scripts/python.exe tests/test_h3_refmod.py
"""
import json
import os
import sys
import tempfile

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


# --- the model: a (h, w, t) reference rides the video clock -----------------------------------
from fizgig.minimax.model import image_position_ids, _video_t_grid, _video_t_spans, _frame_grid  # noqa: E402
fr = _frame_grid(8, 8).shape[0]
pos = image_position_ids(5, 16, 16, 0, refs=[(8, 8, 3)], latent_t=1)
ref = pos[5:5 + 3 * fr]
tg = _video_t_grid(3, 5.0)
ck("video-kind ref: 3 frames x 16 rows, t per frame = _video_t_grid(cursor)",
   ref.shape[0] == 3 * fr and all(torch.allclose(ref[k * fr:(k + 1) * fr, 0],
                                                 torch.full((fr,), float(tg[k]), dtype=torch.float64))
                                  for k in range(3)))
ck("target origin advances by sum(_video_t_spans(3)) after a video-kind ref",
   abs(float(pos[5 + 3 * fr, 0]) - (5.0 + sum(_video_t_spans(3)))) < 1e-9)
pos2 = image_position_ids(5, 16, 16, 0, refs=[(8, 8)], latent_t=1)
ck("a plain (h, w) image ref still advances the cursor by 1.0 (unchanged)",
   abs(float(pos2[5 + fr, 0]) - 6.0) < 1e-9)
import inspect  # noqa: E402
from fizgig.minimax import model as _m  # noqa: E402
ck("forward records (h, w, T) per reference so a multi-frame block gets its rows",
   "ref_shapes.append((z.shape[-2], z.shape[-1], z.shape[2]))" in inspect.getsource(_m.MiniMaxH3DiT.forward))

# --- the builder: pooling, aspect, full canvas ------------------------------------------------
from fizgig.minimax.refmod import (aspect_grid, build_mod, save_refmod, load_refmod, token_count,  # noqa: E402
                                   NODE_META_KEY, NODE_FORMAT_VERSION)
ck("aspect_grid: portrait 42x22 at pool 16 -> (16, 8)", aspect_grid(16, 42 / 22) == (16, 8))
ck("aspect_grid: square stays 16x16", aspect_grid(16, 1.0) == (16, 16))
refs = [("a", torch.randn(24, 42, 22), "clip still"), ("b", torch.randn(24, 40, 24), "photo"),
        ("c", torch.randn(24, 42, 22), "clip still")]
mod, label = build_mod(refs, 16)
ck("pooled mod is [1, 24, T=3, 16, 8] fp32", tuple(mod.shape) == (1, 24, 3, 16, 8) and mod.dtype == torch.float32, tuple(mod.shape))
import torch.nn.functional as _F
ck("pooled = adaptive_avg_pool2d of each source (frame 0)",
   torch.allclose(mod[0, :, 0], _F.adaptive_avg_pool2d(refs[0][1].unsqueeze(0), (16, 8))[0], atol=1e-6))
ck("token_count = T x (h/2)(w/2)", token_count(mod) == 3 * 8 * 4)
full, label_f = build_mod(refs, None)
ck("full mode: every ref on the FIRST ref's even canvas", tuple(full.shape) == (1, 24, 3, 42, 22))
ck("full mode label names pixels (WxH)", "352x672" in label_f, label_f)

# --- the file: the node pack's own reader loads it ---------------------------------------------
with tempfile.TemporaryDirectory() as td:
    p = save_refmod(os.path.join(td, "subj"), mod, name="subj", mode="training", pool=label,
                    optimize_steps=200, source_shape="1x42x22 +1x40x24 +1x42x22",
                    tags=["0 img, 3 clip stills", "fizgig optimised"], extra={"ss_refmod_steps": "200"})
    lat, meta, lsd = load_refmod(p)
    ck("a mod without a LoRA half loads with lora_sd None", lsd is None)
    ck("round trip: latent [1,24,3,16,8] fp16 on disk, meta kind video / latent_t 3 / mode training / v2",
       tuple(lat.shape) == (1, 24, 3, 16, 8) and meta["kind"] == "video" and meta["latent_t"] == 3
       and meta["mode"] == "training" and meta["_format_version"] == NODE_FORMAT_VERSION
       and meta["latent_h"] == 16 and meta["latent_w"] == 8)
    from safetensors import safe_open
    with safe_open(p, "pt") as f:
        hdr = f.metadata()
        ck("header carries refmod_meta JSON + Fizgig's ss_* keys", NODE_META_KEY in hdr and hdr.get("ss_refmod_steps") == "200")
        ck("only the node's `latent` tensor is in the file", list(f.keys()) == ["latent"])
    # the node pack's own class, when its source is around (scratchpad checkout) — the real reader
    node_core = os.environ.get("FIZGIG_REFMOD_NODE_CORE")
    if node_core and os.path.isfile(node_core):
        import importlib.util
        spec = importlib.util.spec_from_file_location("refmod_core", node_core)
        core = importlib.util.module_from_spec(spec)
        sys.modules["refmod_core"] = core          # dataclass resolution needs the module registered
        spec.loader.exec_module(core)
        m = core.H3RefMod.load(p[:-len(".safetensors")])
        blk = m.ref_block(1.0)
        ck("node pack's H3RefMod.load reads it: kind video, token_count 96, ref_block latent_t 3",
           m.kind == "video" and m.token_count == 96 and blk["latent_t"] == 3 and tuple(blk["latent"].shape) == (1, 24, 3, 16, 8))
    else:
        print("skip  node pack reader (set FIZGIG_REFMOD_NODE_CORE=<path to core.py>)")
    # a single-image mod is kind image
    p1 = save_refmod(os.path.join(td, "one"), mod[:, :, :1], name="one", mode="training", pool="1x16x8", optimize_steps=0)
    _, meta1, _ = load_refmod(p1)
    ck("single reference -> kind image, latent_t 1", meta1["kind"] == "image" and meta1["latent_t"] == 1)
    # the superset file: mod + a companion LoRA in ONE safetensors
    fake_lora = {"lora_unet_blocks_0_attn_q.lora_down.weight": torch.randn(2, 5376, dtype=torch.bfloat16),
                 "lora_unet_blocks_0_attn_q.lora_up.weight": torch.zeros(5376, 2, dtype=torch.bfloat16),
                 "lora_unet_blocks_0_attn_q.alpha": torch.tensor(2.0)}
    p2 = save_refmod(os.path.join(td, "pair"), mod, name="pair", mode="training", pool=label, optimize_steps=200,
                     extra={"ss_refmod_lora": "1", "ss_network_dim": "2"}, lora_sd=fake_lora)
    lat2, meta2, lsd2 = load_refmod(p2)
    ck("pair file: latent + 3 lora_unet_* tensors, our reader returns both halves",
       tuple(lat2.shape) == (1, 24, 3, 16, 8) and lsd2 is not None and set(lsd2) == set(fake_lora)
       and meta2["kind"] == "video")
    with safe_open(p2, "pt") as f:
        ck("pair file header still carries refmod_meta (v2) + ss_refmod_lora",
           NODE_META_KEY in f.metadata() and f.metadata().get("ss_refmod_lora") == "1")
    try:
        save_refmod(os.path.join(td, "bad"), mod, name="bad", mode="training", pool=label, optimize_steps=0,
                    lora_sd={"latent": torch.zeros(1)})
        ck("a LoRA dict may not carry non-lora_unet keys", False)
    except ValueError:
        ck("a LoRA dict may not carry non-lora_unet keys", True)
    if node_core and os.path.isfile(node_core):
        m2 = core.H3RefMod.load(p2[:-len(".safetensors")])
        ck("node pack's reader loads the PAIR file unchanged (ignores the LoRA tensors): kind video, 96 tokens",
           m2.kind == "video" and m2.token_count == 96 and tuple(m2.latent.shape) == (1, 24, 3, 16, 8))

# --- the ComfyUI node, with comfy stubbed: conditioning key + block shapes + LoRA patch call ------
import types  # noqa: E402
_stub_calls = {}
_comfy = types.ModuleType("comfy"); _comfy_sd = types.ModuleType("comfy.sd"); _comfy_utils = types.ModuleType("comfy.utils")
_fp = types.ModuleType("folder_paths")
with tempfile.TemporaryDirectory() as td_models:
    _fp.models_dir = td_models
    _fp.add_model_folder_path = lambda *a, **k: _stub_calls.setdefault("folder", a)
    def _load_torch_file(path, safe_load=True, return_metadata=False):
        from safetensors.torch import load_file
        from safetensors import safe_open
        sd = load_file(path)
        with safe_open(path, "pt") as f:
            md = f.metadata()
        return (sd, md) if return_metadata else sd
    _comfy_utils.load_torch_file = _load_torch_file
    def _load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
        _stub_calls["lora"] = (sorted(lora), strength_model, strength_clip)
        return ("patched:" + str(model), clip)
    _comfy_sd.load_lora_for_models = _load_lora_for_models
    _comfy.sd, _comfy.utils = _comfy_sd, _comfy_utils
    _saved_mods = {k: sys.modules.get(k) for k in ("comfy", "comfy.sd", "comfy.utils", "folder_paths")}
    sys.modules.update({"comfy": _comfy, "comfy.sd": _comfy_sd, "comfy.utils": _comfy_utils, "folder_paths": _fp})
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("fizgig_refmod_node", os.path.join(REPO, "comfyui_nodes", "ComfyUI-Fizgig-RefMod", "fizgig_refmod.py"))
        node_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(node_mod)
        ck("node registers the refmods model folder (same folder as the mod pack)", _stub_calls.get("folder", ("",))[0] == "refmods")
        # a pair file in models/refmods
        pair = save_refmod(os.path.join(td_models, "refmods", "pairmod"), mod, name="pairmod", mode="training", pool=label,
                           optimize_steps=200, extra={"ss_network_dim": "2"},
                           lora_sd={"lora_unet_blocks_0_attn_q.lora_down.weight": torch.randn(2, 8, dtype=torch.bfloat16),
                                    "lora_unet_blocks_0_attn_q.lora_up.weight": torch.zeros(8, 2, dtype=torch.bfloat16),
                                    "lora_unet_blocks_0_attn_q.alpha": torch.tensor(2.0)})
        solo = save_refmod(os.path.join(td_models, "refmods", "solomod"), mod[:, :, :1], name="solomod", mode="training", pool="1", optimize_steps=0)
        ck("dropdown lists both files", set(node_mod._list_mods()) == {"pairmod", "solomod"})
        node = node_mod.FizgigH3RefMod()
        cond_in = [[torch.zeros(1, 4, 8), {"minimax_refs": [{"kind": "image", "latent_h": 4, "latent_w": 4, "latent": torch.zeros(1, 24, 1, 4, 4)}]}]]
        m_out, c_out, info = node.apply("MODEL", cond_in, "pairmod", 1.0, 0.8)
        blk = c_out[0][1]["minimax_refs"][-1]
        ck("pair: appends a VIDEO-kind block to minimax_refs after the existing ref (T=3, latent_t/ref_audio_t/audio_latent present)",
           len(c_out[0][1]["minimax_refs"]) == 2 and blk["kind"] == "video" and blk["latent_t"] == 3
           and blk["latent_h"] == 16 and blk["latent_w"] == 8 and blk["ref_audio_t"] == 0 and blk["audio_latent"] is None
           and tuple(blk["latent"].shape) == (1, 24, 3, 16, 8))
        ck("pair: input conditioning untouched (copy, not mutation)", len(cond_in[0][1]["minimax_refs"]) == 1)
        ck("pair: LoRA patched through comfy.sd.load_lora_for_models with the lora_unet_* subset at the given strength",
           m_out == "patched:MODEL" and _stub_calls["lora"][1] == 0.8 and _stub_calls["lora"][2] == 0
           and all(k.startswith("lora_unet_") for k in _stub_calls["lora"][0]) and len(_stub_calls["lora"][0]) == 3)
        ck("info names both halves", "companion LoRA" in info and "3 frame" in info)
        _stub_calls.pop("lora", None)
        m2, c2, info2 = node.apply("MODEL", cond_in, "solomod", 0.5, 1.0)
        b2 = c2[0][1]["minimax_refs"][-1]
        ck("solo: an IMAGE block (no latent_t key), strength 0.5 mixes toward the blurred latent (values changed, shape kept)",
           b2["kind"] == "image" and "latent_t" not in b2 and tuple(b2["latent"].shape) == (1, 24, 1, 16, 8)
           and not torch.equal(b2["latent"], mod[:, :, :1].to(torch.float16).float()))
        ck("solo: no LoRA in the file -> model passed through, no patch call", m2 == "MODEL" and "lora" not in _stub_calls and "no companion LoRA" in info2)
        m3, c3, _ = node.apply("MODEL", cond_in, "pairmod", 0.0, 0.0)
        ck("ref 0 / lora 0: conditioning and model both pass through", c3 is cond_in and m3 == "MODEL")
    finally:
        for k, v in _saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

# --- the GUI: a Base Model entry, two controls, everything else hidden ---------------------------
import tkinter as tk  # noqa: E402
import lora_trainer_gui as g  # noqa: E402
ck("dropdown lists MiniMax H3 RefMod after MiniMax H3",
   g.ARCHITECTURE_LIST.index("MiniMax H3 RefMod") == g.ARCHITECTURE_LIST.index("MiniMax H3") + 1)
cfg = g.ARCHITECTURES["MiniMax H3 RefMod"]
ck("entry is MiniMax (caches, paths) + is_refmod, script minimax_refmod.py, suffix refmod",
   cfg.get("is_minimax") and cfg.get("is_refmod") and cfg["train_script"].endswith("minimax_refmod.py")
   and cfg["lora_name_suffix"] == "refmod")
ck("grid/steps label parsing", g.refmod_grid_value("16×16 (256 tokens, concept-level)") == "16"
   and g.refmod_grid_value("Full reference (recommended — carries the face)") == "full" and g.refmod_grid_value("8×8 (64 tokens, stackable)") == "8"
   and g.refmod_steps_value("200 (recommended)") == "200" and g.refmod_steps_value("0 (encode only — same as the ComfyUI extractor)") == "0")
pr = next(iter(g.REFMOD_BUILT_IN_PRESETS.values()))
ck("the one preset: Full reference, 200 steps, clip still on", pr["MINIMAX_REFMOD_GRID"].startswith("Full")
   and pr["MINIMAX_REFMOD_STEPS"].startswith("200") and pr.get("MINIMAX_CLIP_STILL") is True)
try:
    root = tk.Tk(); root.withdraw()
    app = g.LoRATrainerGUI(root)
    app.save_prefs = lambda *a, **k: None
    app.save_settings = lambda *a, **k: None
    app.architecture_var.set("MiniMax H3 RefMod"); app._on_architecture_selected(); root.update()
    hidden = [k for k in ("training", "memory", "timestep", "optimizer", "scheduler") if not app.collapsible_sections[k].winfo_manager()]
    ck("RefMod: five sections hidden, Output stays, card + Training Base shown",
       len(hidden) == 5 and app.collapsible_sections["output"].winfo_manager()
       and app._refmod_frame.winfo_manager() and app._minimax_base_frame.winfo_manager())
    ck("RefMod preset applied on entry", app.custom_preset_var.get().startswith("✨ MiniMax H3 RefMod"))
    app.settings.update({"DATASET_CONFIG": "d.toml", "LORA_OUTPUT_DIR": "out", "LORA_NAME": "s_refmod", "SEED": "7",
                         "MINIMAX_REFMOD_GRID": "8×8 (64 tokens, stackable)", "MINIMAX_REFMOD_STEPS": "500"})
    app.sample_enabled_var.set(False)
    c = [str(x) for x in app._build_minimax_refmod_command()]
    ck("builder: minimax_refmod.py --grid 8 --steps 500, no LoRA flags",
       c[1].endswith("minimax_refmod.py") and c[c.index("--grid") + 1] == "8" and c[c.index("--steps") + 1] == "500"
       and "--network_dim" not in c and "--learning_rate" not in c)
    ck("Companion LoRA Off -> no --companion_lora_epochs", "--companion_lora_epochs" not in c)
    ck("all three rows on the card; LoRA default Off; every knob registered",
       app._refmod_lora_frame.winfo_manager() and app._refmod_opt_frame.winfo_manager()
       and app.entries["MINIMAX_REFMOD_LORA"].get() == "Off"
       and all(k in app.entries for k in g.REFMOD_DEFAULTS))
    ck("defaults -> the measured recipe on the command line",
       c[c.index("--max_refs") + 1] == "8" and c[c.index("--lr") + 1] == "1e-3" and c[c.index("--pull") + 1] == "2.0"
       and c[c.index("--sigma_min") + 1] == "0.2" and c[c.index("--sigma_max") + 1] == "0.8")
    app.settings.update({"MINIMAX_REFMOD_LORA": "On", "MINIMAX_REFMOD_LORA_RANK": "4", "MINIMAX_REFMOD_LORA_EPOCHS": "3",
                         "MINIMAX_REFMOD_LORA_LR": "1e-4", "MINIMAX_REFMOD_STEPS": "75", "MINIMAX_REFMOD_REFS": "all",
                         "MINIMAX_REFMOD_LR": "2e-3", "MINIMAX_REFMOD_PULL": "0.5", "MINIMAX_REFMOD_SIGMA": "off"})
    c2 = [str(x) for x in app._build_minimax_refmod_command()]
    ck("every GUI knob reaches the command line (typed steps, all refs, LoRA rank/epochs/LR, noise off)",
       c2[c2.index("--steps") + 1] == "75" and c2[c2.index("--max_refs") + 1] == "10000"
       and c2[c2.index("--companion_lora_epochs") + 1] == "3" and c2[c2.index("--companion_lora_rank") + 1] == "4"
       and c2[c2.index("--companion_lora_lr") + 1] == "1e-4" and c2[c2.index("--lr") + 1] == "2e-3"
       and c2[c2.index("--pull") + 1] == "0.5" and c2[c2.index("--sigma_min") + 1] == "-1")
    ck("parsers: legacy 'Rank 2, 2 epochs' label still reads as On; junk numbers fall back to the default",
       g.refmod_lora_on("Rank 2, 2 epochs") and not g.refmod_lora_on("Off") and g.refmod_num("abc", "2.0") == "2.0"
       and g.refmod_num("", "2", int) == "2" and g.refmod_sigma_args("0.3-0.9") == ["--sigma_min", "0.3", "--sigma_max", "0.9"]
       and g.refmod_sigma_args("0.9-0.3")[1] == "0.2")
    app.architecture_var.set("MiniMax H3"); app._on_architecture_selected(); root.update()
    back = [k for k in ("training", "memory", "optimizer", "scheduler") if app.collapsible_sections[k].winfo_manager()]
    ck("back on MiniMax H3: sections return, card hidden", len(back) == 4 and not app._refmod_frame.winfo_manager())
    app.architecture_var.set("Flux 2 Klein Base 9B"); app._on_architecture_selected(); root.update()
    ck("Klein: all five sections packed", all(app.collapsible_sections[k].winfo_manager()
                                            for k in ("training", "memory", "timestep", "optimizer", "scheduler")))
    # the layout bug (Peter, 9 Sep): leaving RefMod re-packed the sections UNDER the button rows.
    # Pin: after RefMod -> H3 and RefMod -> Klein the pack order equals a fresh tab's.
    secs = app.collapsible_sections
    names = {id(v): k for k, v in secs.items()}
    parent = secs["output"].master

    def _order():
        return [names.get(id(w), "other") for w in parent.pack_slaves()]
    fresh_klein = _order()
    app.architecture_var.set("MiniMax H3 RefMod"); app._on_architecture_selected(); root.update()
    app.architecture_var.set("Flux 2 Klein Base 9B"); app._on_architecture_selected(); root.update()
    ck("RefMod -> Klein: section order identical to fresh (sections above the button rows)",
       _order() == fresh_klein, _order())
    app.architecture_var.set("MiniMax H3"); app._on_architecture_selected(); root.update()
    h3_order = _order()
    app.architecture_var.set("MiniMax H3 RefMod"); app._on_architecture_selected(); root.update()
    app.architecture_var.set("MiniMax H3"); app._on_architecture_selected(); root.update()
    ck("RefMod -> MiniMax H3: section order identical to a direct H3 switch", _order() == h3_order, _order())
    ck("sections sit before the first non-section widget after Output",
       _order().index("scheduler") < max(i for i, n in enumerate(_order()) if n == "other"))
    root.destroy()
except tk.TclError as e:
    print(f"skip  GUI (no display: {e})")

print()
print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
