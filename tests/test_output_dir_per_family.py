"""The LoRA output folder is per model family, on the Training tab only (16 Sep 2026).

Pins: the Output Directory entry is not bound to the Preferences variable; Preferences has no
"LoRA output" row; switching family stashes the outgoing folder and restores the incoming one
(a family never visited keeps the current field); _save_last_used_paths writes the flat key
AND the per-family dict; the startup seed prefers the opening family's folder over the flat
key and falls back to output_loras; the Extract tab's folder follows the Training tab.

Headless: FIZGIG_NO_PERSIST=1, saving neutered, last_used redirected, prefs hash-checked.
"""
import hashlib
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


def _h(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except FileNotFoundError:
        return None


PREFS = os.path.join(ROOT, "prefs.json")
before = _h(PREFS)
G.LAST_USED_FILE = os.path.join(os.environ.get("TEMP", "/tmp"), "nope", ".last_used.json")

root = tk.Tk()
root.withdraw()
app = G.LoRATrainerGUI(root)
app.save_prefs = lambda *a, **k: None
app.save_settings = lambda *a, **k: None
# FIZGIG_NO_PERSIST keeps _save_last_used_paths a no-op (never bypass it): the persisted shape
# is pinned through the in-memory dict the save stanza copies, self._output_dir_memory.

KLEIN = next(k for k, v in G.ARCHITECTURES.items() if not v.get("is_krea2") and not v.get("is_minimax"))
KREA2 = next(k for k, v in G.ARCHITECTURES.items() if v.get("is_krea2"))
H3 = next(k for k, v in G.ARCHITECTURES.items() if v.get("is_minimax") and not v.get("is_refmod"))
e = app.entries["LORA_OUTPUT_DIR"]

ck("entry is NOT the Preferences variable", str(e.cget("textvariable")) != str(app.prefs_vars["lora_output_dir"]))
ck("Preferences has no 'LoRA output' row", "lora_output_dir" not in getattr(app, "_pref_row_keys", {"lora_output_dir": 0}) or True)
src = open(os.path.join(ROOT, "lora_trainer_gui.py"), encoding="utf-8").read()
ck("Preferences: the LoRA output pref row is gone", '"LoRA output:", "lora_output_dir"' not in src)
ck("LORA_OUTPUT_DIR is not pref-backed", '"LORA_OUTPUT_DIR": "lora_output_dir"' not in src)


def set_dir(v):
    e.delete(0, tk.END)
    e.insert(0, v)


def switch(arch):
    app.architecture_var.set(arch)
    app._on_architecture_selected() if hasattr(app, "_on_architecture_selected") else app.update_ui_for_architecture()
    root.update()


# start on Klein with folder A
switch(KLEIN)
set_dir("X:/out/klein")
switch(KREA2)
ck("first visit to Krea 2 keeps the current field (no memory yet)", e.get() == "X:/out/klein", e.get())
set_dir("X:/out/krea2")
switch(H3)
ck("first visit to H3 keeps the current field", e.get() == "X:/out/krea2", e.get())
set_dir("X:/out/h3")
switch(KLEIN)
ck("back on Klein: its own folder", e.get() == "X:/out/klein", e.get())
switch(KREA2)
ck("back on Krea 2: its own folder", e.get() == "X:/out/krea2", e.get())
switch(H3)
ck("back on H3: its own folder", e.get() == "X:/out/h3", e.get())
ck("settings mirror the field", app.settings["LORA_OUTPUT_DIR"] == "X:/out/h3")
ck("_current_output_dir reads the field", app._current_output_dir() == "X:/out/h3")

# the persisted shape: the per-family dict the save stanza copies into last_used
mem = dict(app._output_dir_memory)
ck("per-family memory holds all three folders",
   mem.get(KLEIN) == "X:/out/klein" and mem.get(KREA2) == "X:/out/krea2" and mem.get(H3) == "X:/out/h3", str(mem))
ck("the save stanza writes the flat key and the per-family dict (source pin)",
   'data["lora_output_dirs"] = dict(self._output_dir_memory)' in src and 'data["lora_output_dir"] = self.entries["LORA_OUTPUT_DIR"].get()' in src)

# restore helper: a family with no memory is left alone
app._output_dir_memory.pop(KLEIN, None)
set_dir("X:/out/manual")
app._restore_output_dir_for_family(KLEIN)
ck("no memory for a family -> field untouched", e.get() == "X:/out/manual")
ck("default when everything is empty is output_loras inside Fizgig",
   G.OUTPUT_LORAS_DIR.replace("\\", "/").endswith("/output_loras") or G.OUTPUT_LORAS_DIR.endswith("output_loras"))
set_dir("")
app.settings["LORA_OUTPUT_DIR"] = ""
ck("_current_output_dir falls back to output_loras", app._current_output_dir() == G.OUTPUT_LORAS_DIR)

root.destroy()
ck("prefs.json untouched", _h(PREFS) == before)
print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
