"""RefMod Studio tab — headless (FIZGIG_NO_PERSIST=1, prefs/settings saving neutered, the
config files hash-verified untouched, presets redirected to a temp dir).

Pins what fails silently: the tab exists after Repair Studio; rows add / remove / flip to an
A/B axis (scale range); the tokens footer arithmetic and over-cap colour; the job builder's
bundle (copies, retention, step schedule on/off); the ComfyUI readout; bake writes a loadable
mod; curve presets round-trip; state save/restore; the shared clip player opens for BOTH tabs
(Repair Studio's own call unchanged — three sides, no-LoRA hidden; RefMod Studio's — two sides,
its labels, no metrics bar).
"""
import hashlib
import json
import os
import sys
import tempfile

os.environ["FIZGIG_NO_PERSIST"] = "1"
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
import tkinter as tk  # noqa: E402
from PIL import Image  # noqa: E402
import lora_trainer_gui as g  # noqa: E402
from fizgig.minimax import refmod_apply as ra  # noqa: E402
from fizgig.minimax.refmod import save_refmod, load_refmod  # noqa: E402

fails = 0


def ck(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


def _hash(p):
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


GUARDED = [os.path.join(ROOT, "prefs.json"), os.path.join(ROOT, ".last_used.json"),
           os.path.join(ROOT, "Fizgig_train.toml")]
before = {p: _hash(p) for p in GUARDED}
presets_before = sorted(os.listdir(os.path.join(ROOT, "presets"))) if os.path.isdir(os.path.join(ROOT, "presets")) else []

root = tk.Tk()
root.withdraw()
app = g.LoRATrainerGUI(root)
app.save_prefs = lambda *a, **k: None
app.save_settings = lambda *a, **k: None
app._save_last_used_paths = lambda *a, **k: None

tabs = [app.notebook.tab(t, "text") for t in app.notebook.tabs()]
ck("tab present, right after Repair Studio", "RefMod Studio" in tabs and tabs.index("RefMod Studio") == tabs.index("Repair Studio") + 1, str(tabs))
ck("help.json has the tab's key", "refmod_studio" in json.load(open(os.path.join(ROOT, "help.json"), encoding="utf-8"))["youtube_urls"])

with tempfile.TemporaryDirectory() as td:
    app._rms_curve_preset_dir = lambda: (os.makedirs(os.path.join(td, "curves"), exist_ok=True) or os.path.join(td, "curves"))
    app._rms_setup_dir = lambda: td
    torch.manual_seed(1)
    mods = os.path.join(td, "mods")
    os.makedirs(mods)
    save_refmod(os.path.join(mods, "face"), torch.randn(1, 24, 3, 16, 8) * 1.5, name="face", mode="encode", pool="3x16x8",
                optimize_steps=200, description="ginger woman", concept_type="identity")
    save_refmod(os.path.join(mods, "still"), torch.randn(1, 24, 1, 16, 8), name="still", mode="encode", pool="1x16x8", optimize_steps=0)
    save_refmod(os.path.join(mods, "big"), torch.randn(1, 24, 44, 32, 32), name="big", mode="encode", pool="44x32x32", optimize_steps=0)
    app.rms_folder_var.set(mods)
    app._rms_rescan()
    ck("scan finds the three mods", sorted(app._rms_mod_meta) == ["big", "face", "still"], str(list(app._rms_mod_meta)))
    ck("one empty row after build", len(app._rms_rows) == 1)

    row = app._rms_rows[0]
    row["mod_var"].set("face")
    row["value_var"].set(0.8)
    row["copies_var"].set("2")
    app._rms_row_changed(row)
    ck("row info line describes the mod", "video · 3×16×8 · 96 tokens · optimised 200" in row["info"].cget("text"), row["info"].cget("text"))
    ck("tokens footer = 96 x 2 copies", app.rms_tokens_var.get() == "Tokens: 192 / 5 120", app.rms_tokens_var.get())
    ck("plain row scale runs 0..1", float(row["scale"].cget("from")) == 0.0 and row["axis_lbl"].cget("text") == "strength")

    app._rms_add_row()
    r2 = app._rms_rows[1]
    r2["mod_var"].set("face")
    r2["b_var"].set("still")
    r2["value_var"].set(-0.6)
    app._rms_row_changed(r2)
    ck("axis row: scale runs -1..1, label A ◀ 0 ▶ B, value shown signed",
       float(r2["scale"].cget("from")) == -1.0 and r2["axis_lbl"].cget("text") == "A ◀ 0 ▶ B" and r2["value_str"].get() == "-0.60")
    ck("axis row info shows both sides", "A: video" in r2["info"].cget("text") and "B: image" in r2["info"].cget("text"), r2["info"].cget("text"))
    ck("tokens: 192 + A-side 96", app.rms_tokens_var.get() == "Tokens: 288 / 5 120", app.rms_tokens_var.get())
    r2["value_var"].set(0.6)
    app._rms_row_changed(r2)
    ck("axis flipped to B: 192 + 32", app.rms_tokens_var.get() == "Tokens: 224 / 5 120", app.rms_tokens_var.get())

    app._rms_add_row()
    r3 = app._rms_rows[2]
    r3["mod_var"].set("big")
    r3["copies_var"].set("1")
    app._rms_row_changed(r3)
    ck("over the cap turns the footer red", app.rms_tokens_var.get() == "Tokens: 11 488 / 5 120" and app._rms_tokens_lbl.cget("fg") == "#E05050",
       app.rms_tokens_var.get() + " " + app._rms_tokens_lbl.cget("fg"))
    r3["on_var"].set(False)
    app._rms_row_changed(r3)
    ck("disabled row drops out", app.rms_tokens_var.get() == "Tokens: 224 / 5 120")
    app._rms_set_retention(0.0)
    ck("retention 0 -> 0 tokens", app.rms_tokens_var.get() == "Tokens: 0 / 5 120")
    app._rms_set_retention(0.5)
    ck("retention entry mirrors the scale", app.rms_retention_str.get() == "0.50")

    # the job: bundle + schedule
    app.rms_seed_var.set("123")
    app.rms_scramble_var.set("-1")
    job = app._rms_job()
    ck("job bundle: face x2 at 0.8*0.5, still (B) at 0.6*0.5",
       [(n, round(s, 2)) for n, s in job["describe"]] == [("face", 0.4), ("face", 0.4), ("still", 0.3)], str(job["describe"]))
    ck("job latents are the mixed latents in the file's dtype (fp16)", len(job["latents"]) == 3 and job["latents"][0].dtype == torch.float16)
    ck("step curve off -> no schedule", job["schedule"] is None and job["step_curve"] is None)
    app.rms_sc_on_var.set(True)
    app.rms_sc_dir_var.set("concept_at_start")
    job = app._rms_job()
    ck("step curve on -> schedule callable", callable(job["schedule"]) and job["step_curve"][0] == "concept_at_start")
    ck("still by default; baseline key carries the setup", job["frames"] == 1 and job["baseline_key"][1] == 123)
    ck("readout: loader slot, axis row, apply, step curve, hint",
       "mod_1 face   strength_1 0.80   copies_1 2" in job["readout"] and "mod_a_1 face   mod_b_1 still   value_1 +0.60" in job["readout"]
       and "retention 0.50" in job["readout"] and "curve_direction concept_at_start" in job["readout"]
       and "prompt_hint: identity: ginger woman" in job["readout"], job["readout"])
    sweep = app._rms_job({"retention": 1.0, "frames": 1, "label": "retention 1"})
    ck("sweep override patches one dial", [round(s, 2) for _n, s in sweep["describe"]] == [0.8, 0.8, 0.6] and sweep["label"] == "retention 1")
    ck("sweep kinds", [len(app._rms_sweep_items_for(k)) for k in app._RMS_SWEEPS] == [5, 4, 5, 5, 4])

    # hints
    app.rms_prompt_text.delete("1.0", tk.END)
    app.rms_prompt_text.insert("1.0", "a portrait")
    app._rms_add_hints()
    ck("+ mod hints appends the pack's prompt_hint", app.rms_prompt_text.get("1.0", tk.END).strip() == "a portrait, identity: ginger woman")
    app._rms_add_hints()
    ck("…once", app.rms_prompt_text.get("1.0", tk.END).strip() == "a portrait, identity: ginger woman")

    # curve preset round trip
    app.rms_fc_dir_var.set("concept_at_middle"); app.rms_fc_shape_var.set("tanh"); app.rms_fc_value_var.set(0.7)
    app._rms_draw_curves()
    ck("curve canvas draws both curves", len(app._rms_curve_canvas.find_all()) > 20)
    import tkinter.simpledialog as sd
    sd.askstring = lambda *a, **k: "my curve"
    app._rms_curve_preset_save()
    ck("curve preset written", os.path.isfile(os.path.join(td, "curves", "my curve.json")))
    app.rms_fc_dir_var.set("constant"); app.rms_sc_on_var.set(False); app._rms_set_retention(1.0)
    app.rms_curve_preset_var.set("my curve")
    app._rms_curve_preset_load()
    ck("curve preset restores frame + step + retention",
       app._rms_frame_curve() == ("concept_at_middle", "tanh", 0.7) and app.rms_sc_on_var.get() and app._rms_retention() == 0.5)

    # state round trip
    st = app._rms_state()
    ck("state carries rows / curves / retention", len(st["rows"]) == 3 and st["rows"][1]["b"] == "still" and st["retention"] == 0.5
       and st["frame_curve"] == ["concept_at_middle", "tanh", 0.7])
    app._rms_apply_state({"rows": [{"mod": "still", "value": 0.3, "copies": 1}], "retention": 0.9, "seed": "7",
                          "frame_curve": ["concept_at_end", "ease", 1.0], "step_on": False, "prompt": "x", "folder": mods})
    ck("apply_state rebuilds the rows", len(app._rms_rows) == 1 and app._rms_rows[0]["mod_var"].get() == "still" and app._rms_seed() == 7
       and app.rms_tokens_var.get() == "Tokens: 32 / 5 120", app.rms_tokens_var.get())
    app._rms_apply_state(st)
    ck("…and back", len(app._rms_rows) == 3 and app._rms_state()["rows"] == st["rows"])

    # bake (single active non-zero row -> no chooser dialog needed: make it so)
    app._rms_apply_state({"rows": [{"mod": "face", "value": 0.8, "copies": 2}], "retention": 0.7, "seed": "1",
                          "frame_curve": ["concept_at_end", "ease", 1.0], "step_on": True, "prompt": "x", "folder": mods})
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb
    out_path = os.path.join(mods, "face_studio.safetensors")
    fd.asksaveasfilename = lambda *a, **k: out_path
    mb.showinfo = lambda *a, **k: None
    g.messagebox.showinfo = lambda *a, **k: None
    app._rms_bake()
    ck("bake wrote the file", os.path.isfile(out_path))
    if os.path.isfile(out_path):
        z, meta = load_refmod(out_path)
        src = app._rms_latent(app._rms_mod_meta["face"])
        want = ra.ref_block(src, 0.56, ("concept_at_end", "ease", 1.0))
        ck("baked latent = ref_block(strength 0.8 x retention 0.7, frame curve), tags carry the studio line",
           torch.allclose(z, want.float()) and meta["tags"][-1] == "studio: strength 0.56, curve concept_at_end/ease/1.00"
           and meta["kind"] == "video" and meta["description"] == "ginger woman", str(meta.get("tags")))
        ck("baked mod appears in the folder scan", "face_studio" in app._rms_mod_meta)

    # the shared clip player: RefMod Studio's call
    def _clip(colour):
        frames = [Image.new("RGB", (64, 48), colour) for _ in range(3)]
        return {"frames": frames, "middle": frames[1], "wav": None, "steps": 6, "turbo_strength": 0.75, "frames_n": 3, "regime": "custom"}
    app._rms_clips = {"baseline": _clip((10, 10, 10)), "tweaked": _clip((200, 40, 40))}
    app._rms_open_player()
    P = app._repair_player
    ck("RefMod player: two sides, its own clips, no metrics", P is not None and P["sides"] == ["baseline", "tweaked"]
       and P["clips"] is app._rms_clips and P["metrics"] is False)
    ck("RefMod player: our labels", P is not None and P["titles"][0].cget("text").startswith("No mod")
       and P["titles"][1].cget("text").startswith("With mods"), str([t.cget("text") for t in P["titles"]]))
    ck("RefMod player: third pane hidden", P is not None and not P["panes"][2].winfo_manager())
    app._repair_clip_player_swap()
    ck("swap trades the two sides", app._repair_player["sides"] == ["tweaked", "baseline"])
    app._repair_clip_player_close()
    ck("closed", app._repair_player is None)

    # Repair Studio's own call, unchanged
    app._repair_clips = {"baseline": _clip((0, 0, 0)), "tweaked": _clip((0, 0, 255))}
    app._repair_nolora_pending = False
    try:
        app._repair_clip_player_open()
        P = app._repair_player
        ck("Repair player: three side slots, no-LoRA hidden while absent, Repair clips, metrics on",
           P["sides"] == ["nolora", "baseline", "tweaked"] and not P["panes"][0].winfo_manager() and P["clips"] is None
           and P["metrics"] is True and P["titles"][1].cget("text").startswith("Baseline (LoRA at 1.0)"),
           str([t.cget("text") for t in P["titles"]]))
        app._repair_clip_player_close()
    except Exception as e:
        ck("Repair player opens", False, repr(e))

    # unload with nothing loaded is a no-op; tab switch handler runs
    app._rms_unload()
    ck("unload without an engine is quiet", app.rms_engine is None)
    app.notebook.select(app.refmod_studio_tab)
    root.update()
    app.notebook.select(app.repair_studio_tab)
    root.update()
    ck("tab switching with the new tab raises nothing", True)

root.destroy()
after = {p: _hash(p) for p in GUARDED}
ck("config files byte-identical", after == before, str({p: (before[p], after[p]) for p in GUARDED if before[p] != after[p]}))
presets_after = sorted(os.listdir(os.path.join(ROOT, "presets"))) if os.path.isdir(os.path.join(ROOT, "presets")) else []
ck("no preset folders left in the repo", presets_after == presets_before, str(set(presets_after) ^ set(presets_before)))
print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
