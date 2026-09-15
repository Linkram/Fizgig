"""Fine-tune: the text token refiner is FROZEN unless --train_token_refiner (15 Sep 2026).

It used to be activated unconditionally alongside every window (4x the duty cycle of any
block matmul, 1-2% deltas per run). Now the same tick that governs LoRA runs governs FT, off
by default. Pins: the activation is gated on the flag (a fake rotator records the call), the
GUI tick is visible under Fine-tune and emits the flag, and every H3 preset ships it off.
"""
import os
import sys
import types

os.environ["FIZGIG_NO_PERSIST"] = "1"
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

fails = 0


def ck(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


# 1. the trainer's gate — exercised, not grepped: lift the block into a function
src = open(os.path.join(ROOT, "src", "fizgig", "minimax", "trainer.py"), encoding="utf-8").read()
start = src.index('        _refiner = getattr(dit, "token_refiner", None)')
end = src.index("        _cycle_n = len(ft_subset)", start)
block = "\n".join(line[8:] if line.startswith("        ") else line for line in src[start:end].splitlines())


class FakeRotator:
    def __init__(self):
        self.calls = []

    def activate_always(self, prefix, module):
        self.calls.append(prefix)


class FakeLog:
    def __init__(self):
        self.lines = []

    def info(self, msg, *a):
        self.lines.append(msg % a if a else msg)


for flag, want in ((False, []), (True, ["token_refiner"])):
    rot, log = FakeRotator(), FakeLog()
    ns = {"dit": types.SimpleNamespace(token_refiner=object()), "rotator": rot,
          "train_token_refiner": flag, "logger": log}
    exec(block, ns)
    ck(f"train_token_refiner={flag} -> activate_always calls {want}", rot.calls == want, str(rot.calls))
    ck(f"train_token_refiner={flag} -> the log says so",
       any(("TRAINS" if flag else "frozen") in l for l in log.lines), str(log.lines))
rot, log = FakeRotator(), FakeLog()
exec(block, {"dit": types.SimpleNamespace(), "rotator": rot, "train_token_refiner": True, "logger": log})
ck("a base without a refiner: nothing activated, nothing logged", rot.calls == [] and log.lines == [])

# 2. GUI: tick visible under FT, flag emitted when ticked, every preset off
import tkinter as tk  # noqa: E402
import lora_trainer_gui as G  # noqa: E402

root = tk.Tk()
root.withdraw()
g = G.LoRATrainerGUI(root)
g.save_prefs = lambda *a, **k: None
g.save_settings = lambda *a, **k: None
g._save_last_used_paths = lambda *a, **k: None
g.architecture_var.set([k for k in G.ARCHITECTURES if "MiniMax H3" in k and "RefMod" not in k][0])
g.update_ui_for_architecture()
g.minimax_finetune_var.set(True)
g._on_minimax_ft_toggle()
root.update()
ck("refiner tick visible under Fine-tune", g._minimax_refiner_cb.winfo_manager() != "")
ck("hint no longer says LoRA runs only", "LoRA runs only" not in g._minimax_refiner_hint.cget("text")
   and "fine-tune" in g._minimax_refiner_hint.cget("text"))
ck("every H3 preset ships the refiner off",
   all(v.get("MINIMAX_TRAIN_REFINER") is False for v in G.MINIMAX_BUILT_IN_PRESETS.values()))
g.settings["MINIMAX_TRAIN_REFINER"] = False
ck("FT command without the tick carries no --train_token_refiner",
   "--train_token_refiner" not in g._build_minimax_train_command())
g.settings["MINIMAX_TRAIN_REFINER"] = True
cmd = g._build_minimax_train_command()
ck("FT command with the tick carries --train_token_refiner beside --finetune_rotation",
   "--train_token_refiner" in cmd and "--finetune_rotation" in cmd)

# 3. Samples tab note: shown only while H3 + Fine-tune is ticked (previews follow saves)
note = g._samples_ft_note
ck("Samples note shown under H3 fine-tune", note.winfo_manager() != "" and "checkpoint" in note.cget("text")
   and "prompt" in note.cget("text"))
g.minimax_finetune_var.set(False)
g._on_minimax_ft_toggle()
ck("…hidden when Fine-tune is unticked", note.winfo_manager() == "")
g.minimax_finetune_var.set(True)
g._on_minimax_ft_toggle()
g.architecture_var.set("Flux 2 Klein Base 9B")
g.update_ui_for_architecture()
ck("…hidden on another family even with the tick left on", note.winfo_manager() == "")
root.destroy()

print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
