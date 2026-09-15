"""Repair Studio: the Krea 2 'Text fusion x2 / x3 (experimental)' built-in presets.

15 Sep 2026: Peter measured, across several Krea 2 LoRAs, that the four txtfusion blocks at 3x
lift the detail meter (~72 -> ~77) and likeness (2-7 points) with the composition unchanged.
The presets put that one click away. Pins: the Krea 2 dropdown lists Reset All + the two
boosts and nothing Klein-shaped; loading x3 sets exactly the four txtfusion sliders to 3.0
(enabled) and leaves every main block at 1.0; Reset All returns them; Klein's list is untouched
and never shows the boosts; the H3 family is untouched.

Headless: FIZGIG_NO_PERSIST=1, saving neutered, nothing written.
"""
import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import tkinter as tk  # noqa: E402
import lora_trainer_gui as G  # noqa: E402
from fizgig.repair_studio.krea2_blocks import KREA2_TXTFUSION_IDS, all_block_ids_krea2  # noqa: E402

fails = 0


def ck(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


root = tk.Tk()
root.withdraw()
app = G.LoRATrainerGUI(root)
app.save_prefs = lambda *a, **k: None
app.save_settings = lambda *a, **k: None
app._save_last_used_paths = lambda *a, **k: None
app._schedule_preview = lambda *a, **k: None          # no engine, no render

app.repair_family_var.set("krea2")
app._on_repair_family_changed()
names = app._repair_preset_list()
ck("Krea 2 dropdown: Reset All + x2 + x3", names[:3] == ["✨Reset All", "✨Text fusion ×2 (experimental)", "✨Text fusion ×3 (experimental)"], str(names[:4]))
ck("…and none of Klein's semantic presets", not any(n in names for n in ("✨Identity Only", "✨Style+Composition Only", "✨Details Only")))

app._load_repair_preset("✨Text fusion ×3 (experimental)")
mains = [b for b in all_block_ids_krea2() if b not in KREA2_TXTFUSION_IDS]
ck("x3: the four txtfusion sliders read 3.0 and enabled",
   all(app.repair_block_vars[b]["primary_strength"].get() == 3.0 and app.repair_block_vars[b]["primary_enabled"].get()
       for b in KREA2_TXTFUSION_IDS),
   str({b: app.repair_block_vars[b]["primary_strength"].get() for b in KREA2_TXTFUSION_IDS}))
ck("x3: every main block stays at 1.0", all(app.repair_block_vars[b]["primary_strength"].get() == 1.0 for b in mains))
app._load_repair_preset("✨Text fusion ×2 (experimental)")
ck("x2: the four read 2.0", all(app.repair_block_vars[b]["primary_strength"].get() == 2.0 for b in KREA2_TXTFUSION_IDS))
app._load_repair_preset("✨Reset All")
ck("Reset All: back to 1.0", all(app.repair_block_vars[b]["primary_strength"].get() == 1.0 for b in KREA2_TXTFUSION_IDS))

app.repair_family_var.set("klein")
app._on_repair_family_changed()
kn = app._repair_preset_list()
ck("Klein dropdown unchanged: the four semantic presets, no text-fusion entries",
   kn[:4] == ["✨Reset All", "✨Identity Only", "✨Style+Composition Only", "✨Details Only"]
   and not any("Text fusion" in n for n in kn), str(kn[:5]))
app.repair_family_var.set("minimax")
app._on_repair_family_changed()
ck("H3 dropdown has no text-fusion entries", not any("Text fusion" in n for n in app._repair_preset_list()))

readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
ck("README names the presets", "✨Text fusion ×2" in readme and "×3" in readme)

root.destroy()
print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
