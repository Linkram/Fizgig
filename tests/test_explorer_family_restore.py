"""LoRA the Explorer: a remembered family gets its own layout at build (16 Sep 2026).

Peter's screenshot: the app opened with the Explorer on MiniMax H3 (restored from last_used)
and the Klein-only Distilled/Base DiT radio was showing — the build path tested "is Krea 2"
where the click handler tests "not Klein". Pins all three restored families.

Headless: FIZGIG_NO_PERSIST=1; last_used is stubbed per family (nothing written).
"""
import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import tkinter as tk  # noqa: E402
import lora_trainer_gui as G  # noqa: E402

fails = 0


def ck(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


_real_load = G.load_last_used
for fam, dit_shown, ref_strength_shown in (("minimax", False, False), ("krea2", False, False), ("klein", True, True)):
    G.load_last_used = lambda fam=fam: dict(_real_load(), explorer_family=fam)
    root = tk.Tk()
    root.withdraw()
    app = G.LoRATrainerGUI(root)
    app.save_prefs = lambda *a, **k: None
    app.save_settings = lambda *a, **k: None
    app._save_last_used_paths = lambda *a, **k: None
    root.update()
    ck(f"restored {fam}: family var restored", app.explorer_family_var.get() == fam)
    ck(f"restored {fam}: DiT radio {'shown' if dit_shown else 'hidden'}",
       bool(app._explorer_dit_frame.winfo_manager()) == dit_shown, app._explorer_dit_frame.winfo_manager())
    ck(f"restored {fam}: ref Strength {'shown' if ref_strength_shown else 'hidden'}",
       bool(app._explorer_ref_strength_entry.winfo_manager()) == ref_strength_shown)
    root.destroy()
G.load_last_used = _real_load

print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
