"""#136: 'Finish one category early' only reaches the command when the dataset is mixed.

@pauldegroot (15 Sep 2026): a stop epoch saved from a mixed (voice + visual) run came back via
Load Settings From Last Train on a photo-only dataset — the row was hidden, but the flag still
went out and the visuals retired at that epoch (samples froze, timings looked normal).

Headless (FIZGIG_NO_PERSIST=1, saving neutered): three temp folders — mixed, photos only,
voice only — with the same saved stop settings. Only the mixed one emits the flag, and only
the mixed one shows the row.
"""
import os
import sys
import tempfile

os.environ["FIZGIG_NO_PERSIST"] = "1"
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import tkinter as tk  # noqa: E402
from PIL import Image  # noqa: E402
import lora_trainer_gui as G  # noqa: E402

fails = 0


def ck(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


root = tk.Tk()
root.withdraw()
g = G.LoRATrainerGUI(root)
g.save_prefs = lambda *a, **k: None
g.save_settings = lambda *a, **k: None
g._save_last_used_paths = lambda *a, **k: None
g.architecture_var.set("MiniMax H3")
g.update_ui_for_architecture()
g.minimax_finetune_var.set(False)

# what Load Settings From Last Train brings back from a mixed run
g.settings["MIXED_STOP_EPOCH"] = "5"
g.settings["MIXED_STOP_CATEGORY"] = "Photos & clips"
g.settings["MIXED_STOP_MODE"] = "Anchor (keep training at 10% LR)"
g.entries["MIXED_STOP_EPOCH"].delete(0, tk.END)
g.entries["MIXED_STOP_EPOCH"].insert(0, "5")

with tempfile.TemporaryDirectory() as td:
    mixed, photos, voice = (os.path.join(td, n) for n in ("mixed", "photos", "voice"))
    for d in (mixed, photos, voice):
        os.makedirs(d)
    Image.new("RGB", (32, 32)).save(os.path.join(mixed, "a.jpg"))
    open(os.path.join(mixed, "v.wav"), "wb").write(b"\x00" * 64)
    Image.new("RGB", (32, 32)).save(os.path.join(photos, "a.jpg"))
    open(os.path.join(voice, "v.wav"), "wb").write(b"\x00" * 64)

    def probe(folder):
        g.image_folder_var.set(folder)
        g._refresh_audio_only_ui()
        root.update()
        cmd = g._build_minimax_train_command()
        return ("--visual_stop_epoch" in cmd, g._mixed_stop_frame.winfo_manager() != "",
                cmd[cmd.index("--visual_stop_epoch") + 1] if "--visual_stop_epoch" in cmd else None)

    flag, shown, val = probe(mixed)
    ck("mixed dataset: row shown and --visual_stop_epoch 5 emitted", flag and shown and val == "5", (flag, shown, val))
    flag, shown, val = probe(photos)
    ck("photo-only dataset: row hidden and NO stop flag (the #136 case)", not flag and not shown, (flag, shown))
    flag, shown, val = probe(voice)
    ck("voice-only dataset: row hidden and NO stop flag", not flag and not shown, (flag, shown))
    ck("the helper says what the row says", g._minimax_dataset_mixed() is False)
    g.image_folder_var.set(mixed)
    ck("…and True for the mixed folder", g._minimax_dataset_mixed() is True)

root.destroy()
print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
