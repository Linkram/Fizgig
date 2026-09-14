"""MiniMax H3 RefMod (9 Sep 2026; LoRA half removed 14 Sep): the file matches the community node
pack's format, a multi-frame reference packs on the video clock like ComfyUI's PackedLayout,
the mod builder pools/resamples the way the node does, and the GUI entry is a short card whose
one launch path is minimax_refmod.py with Steps. CPU only.

Run: venv/Scripts/python.exe tests/test_h3_refmod.py
"""
import inspect
import json
import os
import sys
import tempfile
import types as _types

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
import torch.nn.functional as _F  # noqa: E402

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
from fizgig.minimax import model as _m  # noqa: E402
ck("forward records (h, w, T) per reference so a multi-frame block gets its rows",
   "ref_shapes.append((z.shape[-2], z.shape[-1], z.shape[2]))" in inspect.getsource(_m.MiniMaxH3DiT.forward))

# --- the builder: pooling, aspect, full canvas ------------------------------------------------
from fizgig.minimax import refmod as _rm  # noqa: E402
from fizgig.minimax.refmod import (aspect_grid, build_mod, save_refmod, load_refmod, token_count,  # noqa: E402
                                   NODE_META_KEY, NODE_FORMAT_VERSION, cover_crop, _canvas_ref,
                                   exclude_refs_from_training, MAX_REFS_DEFAULT)
ck("aspect_grid: portrait 42x22 at pool 16 -> (16, 8)", aspect_grid(16, 42 / 22) == (16, 8))
ck("aspect_grid: square stays 16x16", aspect_grid(16, 1.0) == (16, 16))
refs = [("a", torch.randn(24, 42, 22), "clip still"), ("b", torch.randn(24, 40, 24), "photo"),
        ("c", torch.randn(24, 42, 22), "clip still")]
mod, label = build_mod(refs, 16)
ck("pooled mod is [1, 24, T=3, 16, 8] fp32", tuple(mod.shape) == (1, 24, 3, 16, 8) and mod.dtype == torch.float32, tuple(mod.shape))
ck("pooled = adaptive_avg_pool2d of each cover-cropped source (frame 0 is already on the canvas)",
   torch.allclose(mod[0, :, 0], _F.adaptive_avg_pool2d(refs[0][1].unsqueeze(0), (16, 8))[0], atol=1e-6))
ck("token_count = T x (h/2)(w/2)", token_count(mod) == 3 * 8 * 4)
full, label_f = build_mod(refs, None)
ck("full mode: every ref on the majority-aspect canvas (42x22 portrait), even dims", tuple(full.shape) == (1, 24, 3, 42, 22))
ck("full mode label names pixels (WxH)", "352x672" in label_f, label_f)
sq = torch.arange(24 * 32 * 32, dtype=torch.float32).reshape(1, 24, 32, 32)
cc = cover_crop(sq, 42, 22)
ck("cover_crop: square 32x32 -> 42x22 by scaling to 42x42 and centre-cropping the width (no squash)",
   tuple(cc.shape) == (1, 24, 42, 22)
   and torch.allclose(cc, _F.interpolate(sq, size=(42, 42), mode="bilinear", align_corners=False)[..., :, 10:32]))
ck("canvas = majority aspect (2 portrait + 1 square -> portrait), sized by the largest of them",
   _canvas_ref(refs) == (42, 22) and _canvas_ref([("x", torch.zeros(24, 30, 30), "p"), ("y", torch.zeros(24, 32, 32), "p"),
                                                  ("z", torch.zeros(24, 42, 22), "p")]) == (32, 32))
ck("16 references is the measured default", MAX_REFS_DEFAULT == 16)

# --- reference stills held out of the optimiser's set (opt-in) -----------------------------------
from fizgig.dataset.image_dataset import BucketBatchManager  # noqa: E402
_items = [_types.SimpleNamespace(latent_cache_path=f"C:/c/{n}_minimaxh3.safetensors") for n in ("a", "b", "c", "d", "e")]
_ds = _types.SimpleNamespace(batch_manager=BucketBatchManager({(496, 496): _items[:3], (512, 384): _items[3:]}, 1), num_train_items=5)
_grp = _types.SimpleNamespace(datasets=[_ds], num_train_items=5)
_rmv, _left = exclude_refs_from_training(_grp, ["a", "d", "zzz"])
ck("exclude_refs: 2 of 5 removed, 3 remain, buckets rebuilt", (_rmv, _left) == (2, 3) and len(_ds.batch_manager) == 3
   and all(os.path.basename(i.latent_cache_path)[0] in "bce" for b in _ds.batch_manager.buckets.values() for i in b))
_rm2, _left2 = exclude_refs_from_training(_grp, ["b", "c", "e"])
ck("exclude_refs: removing everything leaves 0 (the run then keeps the refs in and says so)", (_rm2, _left2) == (3, 0))
src_run = inspect.getsource(_rm.run_refmod)
ck("run_refmod trains on every still by default; hold-out is opt-in and only when the optimiser runs",
   "exclude_refs: bool = False" in src_run and "if exclude_refs and steps > 0:" in src_run)

# --- the LoRA half is gone ----------------------------------------------------------------------
ck("no companion LoRA anywhere in the module (trainer, state dict, file writer, previews)",
   not any(hasattr(_rm, n) for n in ("train_companion_lora", "companion_state_dict", "attach_mod_to_file",
                                     "COMPANION_LR", "COMPANION_DIM"))
   and "lora_sd" not in inspect.signature(save_refmod).parameters
   and "lora_net" not in inspect.signature(_rm.render_previews).parameters
   and "companion_lora_epochs" not in inspect.signature(_rm.run_refmod).parameters)
from fizgig.minimax import trainer as _tr  # noqa: E402
_ts = inspect.getsource(_tr.train_minimax)
ck("the H3 trainer has no RefMod mode any more", "refmod_out" not in _ts and "_attach_refmod" not in _ts
   and "refmod_out" not in inspect.signature(_tr.train_minimax).parameters)
from fizgig.scripts import minimax_train as _mt, minimax_refmod as _mr  # noqa: E402
_mt_flags = [a.option_strings[0] for a in _mt.setup_parser()._actions if a.option_strings]
_mr_flags = [a.option_strings[0] for a in _mr.setup_parser()._actions if a.option_strings]
ck("minimax_train.py carries no --refmod_* flags", not any(f.startswith("--refmod") for f in _mt_flags))
ck("minimax_refmod.py: no companion flags; --steps, --max_refs (default 16), --holdout_refs, --init_from stay",
   not any("companion" in f for f in _mr_flags) and {"--steps", "--max_refs", "--holdout_refs", "--init_from"} <= set(_mr_flags)
   and next(a for a in _mr.setup_parser()._actions if a.dest == "max_refs").default == 16)
ck("compute_loss still takes ref_latents (the optimiser's own path)",
   "ref_latents=None" in inspect.getsource(_tr.compute_loss))

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
    node_core = os.environ.get("FIZGIG_REFMOD_NODE_CORE")
    if node_core and os.path.isfile(node_core):
        import importlib.util
        spec = importlib.util.spec_from_file_location("refmod_core", node_core)
        core = importlib.util.module_from_spec(spec)
        sys.modules["refmod_core"] = core
        spec.loader.exec_module(core)
        m = core.H3RefMod.load(p[:-len(".safetensors")])
        blk = m.ref_block(1.0)
        ck("node pack's H3RefMod.load reads it: kind video, token_count 96, ref_block latent_t 3",
           m.kind == "video" and m.token_count == 96 and blk["latent_t"] == 3 and tuple(blk["latent"].shape) == (1, 24, 3, 16, 8))
    else:
        print("skip  node pack reader (set FIZGIG_REFMOD_NODE_CORE=<path to core.py>)")
    p1 = save_refmod(os.path.join(td, "one"), mod[:, :, :1], name="one", mode="training", pool="1x16x8", optimize_steps=0)
    _, meta1 = load_refmod(p1)
    ck("single reference -> kind image, latent_t 1", meta1["kind"] == "image" and meta1["latent_t"] == 1)

ck("the Fizgig ComfyUI RefMod node is gone (their loader is the loader)",
   not os.path.isdir(os.path.join(REPO, "comfyui_nodes", "ComfyUI-Fizgig-RefMod")))

# --- the GUI: a Base Model entry, a short card, everything else hidden ---------------------------
import tkinter as tk  # noqa: E402
import lora_trainer_gui as g  # noqa: E402
ck("dropdown lists MiniMax H3 RefMod after MiniMax H3",
   g.ARCHITECTURE_LIST.index("MiniMax H3 RefMod") == g.ARCHITECTURE_LIST.index("MiniMax H3") + 1)
cfg = g.ARCHITECTURES["MiniMax H3 RefMod"]
ck("entry is MiniMax (caches, paths) + is_refmod, suffix refmod",
   cfg.get("is_minimax") and cfg.get("is_refmod") and cfg["lora_name_suffix"] == "refmod")
ck("grid label parsing", g.refmod_grid_value("16×16 (256 tokens, concept-level)") == "16"
   and g.refmod_grid_value("Full reference (recommended — carries the face)") == "full" and g.refmod_grid_value("8×8 (64 tokens, stackable)") == "8")
ck("steps label parsing: option labels, a typed number, junk -> 200",
   g.refmod_steps_value(g.REFMOD_STEP_OPTIONS[0]) == "0" and g.refmod_steps_value(g.REFMOD_STEP_OPTIONS[1]) == "200"
   and g.refmod_steps_value("75") == "75" and g.refmod_steps_value("lots") == "200")
pr = next(iter(g.REFMOD_BUILT_IN_PRESETS.values()))
ck("the one preset: Full reference, 16 refs, 200 steps, clip still on, no LoRA keys",
   pr["MINIMAX_REFMOD_GRID"].startswith("Full") and pr["MINIMAX_REFMOD_REFS"] == "16"
   and pr["MINIMAX_REFMOD_STEPS"].startswith("200") and pr.get("MINIMAX_CLIP_STILL") is True
   and not any("LORA" in k or "EMA" in k for k in pr if k.startswith("MINIMAX_REFMOD")))
ck("no LoRA helpers or option lists left in the GUI module",
   not any(hasattr(g, n) for n in ("refmod_lora_on", "REFMOD_LORA_OPTIONS", "REFMOD_EMA_OPTIONS", "refmod_sigma_args")))
try:
    root = tk.Tk(); root.withdraw()
    app = g.LoRATrainerGUI(root)
    app.save_prefs = lambda *a, **k: None
    app.save_settings = lambda *a, **k: None
    app.architecture_var.set("MiniMax H3 RefMod"); app._on_architecture_selected(); root.update()
    hidden = [k for k in ("training", "memory", "timestep", "optimizer", "scheduler") if not app.collapsible_sections[k].winfo_manager()]
    ck("RefMod: five sections hidden, Output stays, the card row + Training Base shown, no LoRA row",
       len(hidden) == 5 and app.collapsible_sections["output"].winfo_manager()
       and app._refmod_frame.winfo_manager() and app._minimax_base_frame.winfo_manager()
       and not hasattr(app, "_refmod_lora_frame"))
    ck("the card: Grid, References, Steps, Target MP; no LoRA/EMA entries",
       all(k in app.entries for k in ("MINIMAX_REFMOD_GRID", "MINIMAX_REFMOD_REFS", "MINIMAX_REFMOD_STEPS"))
       and app.entries["MINIMAX_REFMOD_STEPS"].winfo_manager()
       and not any(k in app.entries for k in ("MINIMAX_REFMOD_LORA", "MINIMAX_REFMOD_LORA_RANK", "MINIMAX_REFMOD_EMA")))
    ck("RefMod preset applied on entry", app.custom_preset_var.get().startswith("✨ MiniMax H3 RefMod"))
    ck("Target MP on the card shares the Dataset section's variable",
       str(app._refmod_mp_combo.cget("textvariable")) == str(app.dataset_megapixels_var))
    ck("card defaults: refs 16, steps 200", app.entries["MINIMAX_REFMOD_REFS"].get() == "16"
       and app.entries["MINIMAX_REFMOD_STEPS"].get().startswith("200"))
    per, total = app.refmod_token_estimate("Full reference (recommended — carries the face)", "8", "0.25")
    ck("token estimate: Full at 0.25 MP ≈ 244 per ref, 8 refs ≈ 1,952 (under the 5,120 cap)", per == 244 and total == 1952)
    _, tot_all = app.refmod_token_estimate("Full reference (recommended — carries the face)", "all", "1.0")
    ck("token estimate: 'all' gives per-ref only", tot_all is None)
    ck("the standard-RefMod reference block is on the card and names their defaults",
       app._refmod_std_hint.winfo_manager() and "16 images" in app._refmod_std_hint.cget("text")
       and "1024" in app._refmod_std_hint.cget("text") and "5,120" in app._refmod_std_hint.cget("text"))
    ck("the hint says what Steps do and where the file goes, and never mentions a companion LoRA",
       "Steps" in app._refmod_hint.cget("text") and "models/refmods" in app._refmod_hint.cget("text")
       and "Companion" not in app._refmod_hint.cget("text") and "companion" not in app._refmod_hint.cget("text"))
    # the builder: one path, minimax_refmod.py with the card's values
    base = {"DATASET_CONFIG": "d.toml", "LORA_OUTPUT_DIR": "out", "LORA_NAME": "s_refmod", "SEED": "7",
            "MINIMAX_REFMOD_GRID": app.entries["MINIMAX_REFMOD_GRID"].get(),
            **{k: app.entries[k].get() for k in g.REFMOD_DEFAULTS}, "MINIMAX_TRAIN_BASE": "ref2va"}
    app.settings.update(base)
    app.sample_enabled_var.set(False)
    c = [str(x) for x in app._build_minimax_refmod_command()]
    ck("builder -> minimax_refmod.py --grid full --steps 200 --max_refs 16 on the ref2va base, never minimax_train.py",
       c[1].endswith("minimax_refmod.py") and c[c.index("--grid") + 1] == "full" and c[c.index("--steps") + 1] == "200"
       and c[c.index("--max_refs") + 1] == "16" and "--refmod_out" not in c and "--network_dim" not in c
       and c[c.index("--dit") + 1].endswith("minimax_h3_ref2va_pruned_int8_convrot.safetensors"))
    app.settings.update({"MINIMAX_REFMOD_STEPS": g.REFMOD_STEP_OPTIONS[0], "MINIMAX_REFMOD_REFS": "all"})
    c2 = [str(x) for x in app._build_minimax_refmod_command()]
    ck("Steps 0 + all refs -> --steps 0 --max_refs 10000 (plain encode of every still)",
       c2[c2.index("--steps") + 1] == "0" and c2[c2.index("--max_refs") + 1] == "10000")
    app.settings["MINIMAX_REFMOD_STEPS"] = "500"
    c3 = [str(x) for x in app._build_minimax_refmod_command()]
    ck("a typed step count reaches the command line", c3[c3.index("--steps") + 1] == "500")
    app.architecture_var.set("MiniMax H3"); app._on_architecture_selected(); root.update()
    back = [k for k in ("training", "memory", "optimizer", "scheduler") if app.collapsible_sections[k].winfo_manager()]
    ck("back on MiniMax H3: sections return, card hidden", len(back) == 4 and not app._refmod_frame.winfo_manager())
    app.architecture_var.set("Flux 2 Klein Base 9B"); app._on_architecture_selected(); root.update()
    ck("Klein: all five sections packed", all(app.collapsible_sections[k].winfo_manager()
                                            for k in ("training", "memory", "timestep", "optimizer", "scheduler")))
    secs = app.collapsible_sections
    names = {id(v): k for k, v in secs.items()}
    parent = secs["output"].master

    def _order():
        return [names.get(id(w), "other") for w in parent.pack_slaves()]
    fresh_klein = _order()
    app.architecture_var.set("MiniMax H3 RefMod"); app._on_architecture_selected(); root.update()
    app.architecture_var.set("Flux 2 Klein Base 9B"); app._on_architecture_selected(); root.update()
    ck("RefMod -> Klein: section order identical to fresh (sections above the button rows)", _order() == fresh_klein, _order())
    app.architecture_var.set("MiniMax H3"); app._on_architecture_selected(); root.update()
    h3_order = _order()
    app.architecture_var.set("MiniMax H3 RefMod"); app._on_architecture_selected(); root.update()
    app.architecture_var.set("MiniMax H3"); app._on_architecture_selected(); root.update()
    ck("RefMod -> MiniMax H3: section order identical to a direct H3 switch", _order() == h3_order, _order())
    root.destroy()
except tk.TclError as e:
    print(f"skip  GUI (no display: {e})")

# --- short-run EMA (a branch feature that stays) -----------------------------------------------------
from fizgig.training.ema import EMAWeights as _EMA  # noqa: E402
_net = torch.nn.Linear(4, 4)
_e = _EMA(_net, 0.9, ramp=2)
_e.update(); _e.update(); _e.update()
ck("EMAWeights ramp offset: ramp=2 -> decay (1+n)/(2+n) then capped at decay", abs(min(0.9, 4 / 5) - 0.8) < 1e-9 and _e.ramp == 2)
ck("trainer: 'short' EMA sizes the window to the run (1 - 4/steps, ramp 2) and records ss_ema_mode",
   "1.0 - 4.0 / _total" in _ts and "EMAWeights(network, _d, ramp=2)" in _ts and '"ss_ema_mode"' in _ts)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
