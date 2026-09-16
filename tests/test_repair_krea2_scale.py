"""Repair Studio on Krea 2: the 'at strength' load multiplier, as on MiniMax H3 (16 Sep 2026).

Pins: the boxes show under Krea 2 and H3, not Klein; editing the primary strength under Krea 2
lands in repair_state and re-renders; the Krea 2 engine multiplies every block slider by the
load strength (slider 0.5 at strength 0.7 -> module multiplier 0.35, donor likewise); the
baseline is rendered at the load strength and re-renders when it changes; the bake path never
reads the load strength (the saved file keeps its original scale).

Headless: FIZGIG_NO_PERSIST=1, saving neutered; the engine is exercised with fake networks.
"""
import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import tkinter as tk  # noqa: E402
import lora_trainer_gui as G  # noqa: E402
from fizgig.repair_studio.state import SliderState  # noqa: E402
from fizgig.repair_studio import krea2_engine as KE  # noqa: E402

fails = 0


def ck(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


# ── engine: slider x load strength, baseline at the load strength ─────────────────────────
class FakeNet:
    def __init__(self):
        self.mult, self.enabled = {}, {}

    def set_module_enabled_by_pattern(self, pat, on, target="unet"):
        self.enabled[pat] = bool(on)

    def set_module_multiplier_by_pattern(self, pat, m, target="unet"):
        self.mult[pat] = float(m)


eng = KE.__dict__[next(n for n in dir(KE) if n.endswith("RepairEngine") and n.startswith("Krea"))].__new__(
    KE.__dict__[next(n for n in dir(KE) if n.endswith("RepairEngine") and n.startswith("Krea"))])
eng.primary_network, eng.donor_network = FakeNet(), FakeNet()
st = SliderState.default_krea2()
st.blocks["block_3"].primary_strength = 0.5
st.blocks["block_3"].donor_strength = 2.0
st.primary_scale, st.donor_scale = 0.7, 0.25
eng.apply_state(st)
from fizgig.repair_studio.krea2_blocks import block_regex_krea2  # noqa: E402
p3 = block_regex_krea2("block_3")
p0 = block_regex_krea2("block_0")
ck("primary: slider 0.5 x strength 0.7 -> 0.35", abs(eng.primary_network.mult[p3] - 0.35) < 1e-9, eng.primary_network.mult.get(p3))
ck("primary: untouched block 1.0 x 0.7 -> 0.7", abs(eng.primary_network.mult[p0] - 0.7) < 1e-9)
ck("donor: 2.0 x 0.25 -> 0.5", abs(eng.donor_network.mult[p3] - 0.5) < 1e-9)
ck("all 32 blocks pushed", len(eng.primary_network.mult) == 32)

# baseline: keyed on the load strength, rendered from a default state AT that strength
eng.primary_path = "p.safetensors"
eng._baseline_cache_key, eng._baseline_cache_image = None, None
seen = []
eng.generate_preview = lambda state, **k: (seen.append((state.primary_scale, state.donor_scale,
                                                        all(b.primary_strength == 1.0 for b in state.blocks.values())))
                                           or f"img{len(seen)}")
a = eng.generate_baseline(st)
b = eng.generate_baseline(st)
ck("baseline rendered once for the same strength (cached)", a == b == "img1" and len(seen) == 1)
ck("baseline state = every slider 1.0 at the LOAD strength", seen[0] == (0.7, 0.25, True), str(seen[0]))
st.primary_scale = 1.0
c = eng.generate_baseline(st)
ck("changing the load strength re-renders the baseline", c == "img2" and seen[1][0] == 1.0)

# the bake never reads the load strength
bake_src = open(os.path.join(ROOT, "src", "fizgig", "repair_studio", "bake.py"), encoding="utf-8").read()
ck("bake.py never applies primary_scale / donor_scale (the file keeps its original scale)",
   "primary_scale" not in bake_src and "donor_scale" not in bake_src)

# ── GUI: the boxes per family, the edit lands in the state ───────────────────────────────
root = tk.Tk()
root.withdraw()
app = G.LoRATrainerGUI(root)
app.save_prefs = lambda *a, **k: None
app.save_settings = lambda *a, **k: None
app._save_last_used_paths = lambda *a, **k: None
rendered = []
app._on_preview_param_changed = lambda *a, **k: rendered.append(1)


def shown():
    return [bool(spin.winfo_manager()) for _l, spin in app._repair_scale_widgets]


app.repair_family_var.set("krea2")
app._on_repair_family_changed()
root.update()
ck("Krea 2: both 'at strength' boxes shown", shown() == [True, True], str(shown()))
app.repair_primary_scale_var.set("0.8")
app._on_repair_scale_changed()
ck("Krea 2: editing the primary strength lands in repair_state and re-renders",
   abs(app.repair_state.primary_scale - 0.8) < 1e-9 and rendered == [1])
app._on_repair_scale_changed()
ck("…unchanged value does not re-render", rendered == [1])
app.repair_family_var.set("minimax")
app._on_repair_family_changed()
root.update()
ck("H3: boxes shown (unchanged)", shown() == [True, True])
app.repair_family_var.set("klein")
app._on_repair_family_changed()
root.update()
ck("Klein: boxes hidden", shown() == [False, False])
app.repair_primary_scale_var.set("0.3")
n0 = len(rendered)
app._on_repair_scale_changed()
ck("Klein: the handler is inert", len(rendered) == n0)
root.destroy()

print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
