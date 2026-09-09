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
    lat, meta = load_refmod(p)
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
    _, meta1 = load_refmod(p1)
    ck("single reference -> kind image, latent_t 1", meta1["kind"] == "image" and meta1["latent_t"] == 1)

# --- the GUI: a Base Model entry, two controls, everything else hidden ---------------------------
import tkinter as tk  # noqa: E402
import lora_trainer_gui as g  # noqa: E402
ck("dropdown lists MiniMax H3 RefMod after MiniMax H3",
   g.ARCHITECTURE_LIST.index("MiniMax H3 RefMod") == g.ARCHITECTURE_LIST.index("MiniMax H3") + 1)
cfg = g.ARCHITECTURES["MiniMax H3 RefMod"]
ck("entry is MiniMax (caches, paths) + is_refmod, script minimax_refmod.py, suffix refmod",
   cfg.get("is_minimax") and cfg.get("is_refmod") and cfg["train_script"].endswith("minimax_refmod.py")
   and cfg["lora_name_suffix"] == "refmod")
ck("grid/steps label parsing", g.refmod_grid_value("16×16 (256 tokens, recommended)") == "16"
   and g.refmod_grid_value("Full reference (encode canvas)") == "full" and g.refmod_grid_value("8×8 (64 tokens, stackable)") == "8"
   and g.refmod_steps_value("200 (recommended)") == "200" and g.refmod_steps_value("0 (encode only — same as the ComfyUI extractor)") == "0")
pr = next(iter(g.REFMOD_BUILT_IN_PRESETS.values()))
ck("the one preset: 16x16, 200 steps, clip still on", pr["MINIMAX_REFMOD_GRID"].startswith("16×16")
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
    app.architecture_var.set("MiniMax H3"); app._on_architecture_selected(); root.update()
    back = [k for k in ("training", "memory", "optimizer", "scheduler") if app.collapsible_sections[k].winfo_manager()]
    ck("back on MiniMax H3: sections return, card hidden", len(back) == 4 and not app._refmod_frame.winfo_manager())
    app.architecture_var.set("Flux 2 Klein Base 9B"); app._on_architecture_selected(); root.update()
    ck("Klein: all five sections packed", all(app.collapsible_sections[k].winfo_manager()
                                            for k in ("training", "memory", "timestep", "optimizer", "scheduler")))
    root.destroy()
except tk.TclError as e:
    print(f"skip  GUI (no display: {e})")

print()
print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
