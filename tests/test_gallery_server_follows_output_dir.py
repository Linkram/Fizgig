"""The sample gallery's HTTP server follows the CURRENT LoRA output folder.

13 Sep 2026 (Peter): start a Krea 2 run, stop it, change the LoRA output folder, start a
MiniMax run — the gallery showed no previews. The server starts once per session and had
captured the first folder; the watcher wrote files.json into the new one. Now both the
samples folder and the /loras/ folder resolve per request from the live settings.

Headless: FIZGIG_NO_PERSIST=1, prefs/settings saving neutered, nothing written outside temp.
"""
import json
import os
import sys
import tempfile
import urllib.request

os.environ["FIZGIG_NO_PERSIST"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tkinter as tk  # noqa: E402
import lora_trainer_gui as g  # noqa: E402

fails = 0


def ck(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, r.read().decode("utf-8")


root = tk.Tk()
root.withdraw()
app = g.LoRATrainerGUI(root)
app.save_prefs = lambda *a, **k: None
app.save_settings = lambda *a, **k: None

with tempfile.TemporaryDirectory() as tmp:
    a = os.path.join(tmp, "run_a")
    b = os.path.join(tmp, "run_b")
    for d, tag in ((a, "A"), (b, "B")):
        os.makedirs(os.path.join(d, "sample"), exist_ok=True)
        with open(os.path.join(d, "sample", "files.json"), "w", encoding="utf-8") as f:
            json.dump([f"{tag}.png"], f)
        with open(os.path.join(d, f"lora_{tag}-000001.safetensors"), "wb") as f:
            f.write(tag.encode())

    app.settings["LORA_OUTPUT_DIR"] = a
    app.start_gallery_server()
    port = app.gallery_server_port
    ck("server started", bool(port))
    st, body = get(port, "/files.json")
    ck("serves the first run's samples", st == 200 and json.loads(body) == ["A.png"], body)
    st, body = get(port, "/loras/lora_A-000001.safetensors")
    ck("serves the first run's checkpoint", st == 200 and body == "A")

    # the output folder changes mid-session (a new run elsewhere), server NOT restarted
    app.settings["LORA_OUTPUT_DIR"] = b
    ck("server object unchanged (no restart)", app.gallery_server_port == port)
    st, body = get(port, "/files.json")
    ck("now serves the NEW run's samples", st == 200 and json.loads(body) == ["B.png"], body)
    st, body = get(port, "/loras/lora_B-000001.safetensors")
    ck("and the NEW run's checkpoint", st == 200 and body == "B")
    try:
        st, _ = get(port, "/loras/lora_A-000001.safetensors")
    except urllib.error.HTTPError as e:
        st = e.code
    ck("the old run's checkpoint is no longer reachable", st == 404, str(st))

    app.stop_gallery_server()

root.destroy()
print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
