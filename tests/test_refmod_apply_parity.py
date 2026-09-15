"""RefMod Studio maths == the community node pack's maths, measured.

Every number the Studio produces for a reference latent (frame curve, step curve, strength
lerp, retention, copies, axis, scramble) is compared against the pack's own core.py, loaded
from the mirrored copy (set FIZGIG_REFMOD_NODE_CORE=<path to core.py>; without it the
curve/ref_block comparisons are skipped and only the self-consistency checks run).

The one deliberate difference is pinned as such: the pack's constant+linear@1.0 frame curve
skips the flat strength on a video mod (their ref_block bug); ours applies it.
"""
import importlib.util
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.minimax import refmod_apply as ra  # noqa: E402
from fizgig.minimax.refmod import blur_latent  # noqa: E402

fails = 0


def ck(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


core = None
node_core = os.environ.get("FIZGIG_REFMOD_NODE_CORE")
if node_core and os.path.isfile(node_core):
    spec = importlib.util.spec_from_file_location("refmod_core", node_core)
    core = importlib.util.module_from_spec(spec)
    sys.modules["refmod_core"] = core
    spec.loader.exec_module(core)
else:
    print("skip  node pack core (set FIZGIG_REFMOD_NODE_CORE=<path to core.py>)")

torch.manual_seed(0)
T_SET = (1, 5, 16, 44)
V_SET = (0.3, 1.0)
XS = [i / 40 for i in range(41)]

# ── curves: every shape x direction x t x value ──────────────────────────────────────────────
if core is not None:
    ck("direction / shape tables match", tuple(core.CURVE_DIRECTIONS) == ra.CURVE_DIRECTIONS
       and tuple(core.CURVE_SHAPES) == ra.CURVE_SHAPES)
    bad = []
    for shape in ra.CURVE_SHAPES:
        for x in XS:
            if abs(core._ease(shape, x) - ra.ease(shape, x)) > 1e-12:
                bad.append((shape, x))
    ck("ease(): all 11 shapes at 41 points", not bad, str(bad[:3]))
    bad = []
    n = 0
    for d in ra.CURVE_DIRECTIONS:
        for s in ra.CURVE_SHAPES:
            for v in V_SET:
                spec_ = (d, s, v)
                for x in XS:
                    n += 1
                    if abs(core.curve_value_at(spec_, x) - ra.curve_value_at(spec_, x)) > 1e-12:
                        bad.append((spec_, x))
                for t in T_SET:
                    theirs = core.curve_strengths(spec_, t)
                    ours = ra.curve_strengths(spec_, t)
                    if (theirs is None) != (ours is None):
                        bad.append((spec_, t, "None-ness"))
                    elif theirs is not None and any(abs(a - b) > 1e-12 for a, b in zip(theirs, ours)):
                        bad.append((spec_, t))
    ck(f"curve_value_at + curve_strengths: {len(ra.CURVE_DIRECTIONS) * len(ra.CURVE_SHAPES) * len(V_SET)} "
       f"specs x 41 points x t in {T_SET}", not bad, str(bad[:3]))
    ck("unknown direction -> None / 1.0 like theirs",
       ra.curve_strengths(("sideways", "ease", 1.0), 16) is None
       and core.curve_strengths(("sideways", "ease", 1.0), 16) is None
       and ra.curve_value_at(("sideways", "ease", 1.0), 0.5) == 1.0)

# ── ref_block: image + video, fp16 + fp32, curve and flat ───────────────────────────────────
def _mod(t, h=16, w=8, dtype=torch.float32):
    return (torch.randn(1, 24, t, h, w) * 1.7).to(dtype)


if core is not None:
    bad = []
    for t in (1, 16):
        for dtype in (torch.float16, torch.float32):
            z = _mod(t, dtype=dtype)
            kind = "video" if t > 1 else "image"
            m = core.H3RefMod(name="x", kind=kind, latent_h=16, latent_w=8, latent_t=t, latent=z)
            for strength in (1.0, 0.56, 0.15):
                for curve in (None, ("concept_at_end", "ease", 1.0), ("concept_at_middle", "sigmoid", 0.3),
                              ("constant", "dip", 1.0)):
                    theirs = m.ref_block(strength, curve=curve)
                    ours = ra.ref_block(z, strength, curve)
                    if theirs["latent"].dtype != ours.dtype or not torch.equal(theirs["latent"], ours):
                        diff = (theirs["latent"].float() - ours.float()).abs().max().item()
                        bad.append((t, str(dtype), strength, curve, diff))
    ck("ref_block bit-identical to theirs: t in (1,16) x fp16/fp32 x 3 strengths x 4 curves",
       not bad, str(bad[:3]))
    z = _mod(16)
    m = core.H3RefMod(name="x", kind="video", latent_h=16, latent_w=8, latent_t=16, latent=z)
    theirs = m.ref_block(0.4, curve=("constant", "linear", 1.0))["latent"]
    ours = ra.ref_block(z, 0.4, ("constant", "linear", 1.0))
    ck("DELIBERATE difference: constant+linear@1.0 — theirs ignores the 0.4 strength (full latent), ours applies it",
       torch.equal(theirs, z) and torch.equal(ours, ra.ref_block(z, 0.4, None)))
    ck("strength <= 0 -> no block, both", m.ref_block(0.0) is None and ra.ref_block(z, 0.0) is None)
    ck("strength 1.0, no curve -> the stored latent untouched, both",
       torch.equal(m.ref_block(1.0)["latent"], z) and ra.ref_block(z, 1.0) is z)

# self-consistency (runs without the pack)
z = _mod(5)
half = ra.ref_block(z, 0.5)
ck("strength 0.5 is the midpoint between the latent and its blur",
   torch.allclose(half, 0.5 * z + 0.5 * blur_latent(z), atol=1e-6))
zi = _mod(1)
ck("image mod: curve path == flat path", torch.equal(ra.ref_block(zi, 0.7, ("concept_at_end", "ease", 1.0)),
                                                    ra.ref_block(zi, 0.7, None)))

# ── bundle: rows, copies, axis, retention, scramble ─────────────────────────────────────────
za, zb, zc = _mod(3), _mod(1), _mod(2)
ma, mb, mc = {"name": "a", "description": "ginger woman", "concept_type": "identity"}, {"name": "b"}, {"name": "c"}
rows = [ra.ModRow(za, ma, value=0.8, copies=2, name="a"),
        ra.ModRow(zb, mb, value=1.0, copies=1, name="b"),
        ra.ModRow(za, ma, value=-0.6, copies=1, name="a", b_latent=zc, b_meta=mc, b_name="c"),   # axis -> A
        ra.ModRow(za, ma, value=+0.25, copies=1, name="a", b_latent=zc, b_meta=mc, b_name="c"),  # axis -> B
        ra.ModRow(zb, mb, value=0.9, copies=3, name="b", enabled=False),
        ra.ModRow(zc, mc, value=0.0, copies=1, name="c")]
loads = ra.loads_from_rows(rows)
ck("loader order + copies + axis sides + disabled/zero rows skipped",
   [(n, round(s, 2)) for _z, _m, s, n in loads] == [("a", 0.8), ("a", 0.8), ("b", 1.0), ("a", 0.6), ("c", 0.25)])
ck("tokens: copies x per-mod tokens, disabled rows 0",
   ra.bundle_tokens(rows) == 2 * 3 * 32 + 1 * 32 + 1 * 3 * 32 + 1 * 2 * 32)

lat, desc = ra.build_bundle(rows, retention=0.5, curve=None, scramble_seed=-1)
ck("retention folds into every strength", [round(s, 3) for _n, s in desc] == [0.4, 0.4, 0.5, 0.3, 0.125])
ck("bundle latents are the strength-lerped rows", torch.allclose(lat[2], ra.ref_block(zb, 0.5)) and torch.allclose(lat[4], ra.ref_block(zc, 0.125)))
lat0, _ = ra.build_bundle(rows, retention=0.0)
ck("retention 0 -> nothing injected", lat0 == [])


def _their_scramble(items, seed):
    items = list(items)
    if int(seed) >= 0 and len(items) > 1:
        rng = random.Random(int(seed))
        rng.shuffle(items)
        keep = rng.randint(max(1, len(items) // 2), len(items))
        items = items[:keep]
    return items


ok = all(ra.scramble(list(range(5)), s) == _their_scramble(list(range(5)), s) for s in range(0, 40))
ck("scramble: same shuffle + same kept prefix as the pack for seeds 0..39", ok)
ck("scramble off for seed -1 and for one entry", ra.scramble([1, 2, 3], -1) == [1, 2, 3] and ra.scramble([7], 3) == [7])

# ── step schedule: their wrapper formula ────────────────────────────────────────────────────
spec_ = ("concept_at_end", "ease", 1.0)
sched = ra.step_schedule(spec_, [za, zc])
bad = []
for sigma in (1.0, 0.8, 0.5, 0.2, 0.0):
    got = sched(0, 6, sigma)
    s = (core.curve_value_at(spec_, 1.0 - sigma) if core is not None else ra.curve_value_at(spec_, 1.0 - sigma))
    for g, p in zip(got, (za, zc)):
        want = p if s >= 1.0 else s * p + (1.0 - s) * blur_latent(p)
        if not torch.allclose(g, want, atol=1e-6):
            bad.append(sigma)
ck("step schedule: s = curve(1 - sigma), pristine at s >= 1, blur-mix below", not bad, str(bad))
ck("flat step curve -> no schedule", ra.step_schedule(("constant", "linear", 1.0), [za]) is None
   and ra.step_schedule(None, [za]) is None)
ck("first step (sigma 1) under concept_at_end is the pristine latent; last step is fully blurred",
   sched(0, 6, 1.0)[0] is za and torch.allclose(sched(5, 6, 0.0)[0], blur_latent(za), atol=1e-6))

# ── readout + hint + bake ───────────────────────────────────────────────────────────────────
txt = ra.comfy_readout(rows, retention=0.7, frame_curve=("concept_at_end", "ease", 1.0), scramble_seed=-1,
                       step_curve=("concept_at_start", "tanh", 0.5), step_on=True)
ck("readout lists loader slots, axis rows, apply and step curve",
   "mod_1 a   strength_1 0.80   copies_1 2" in txt and "mod_a_1 a   mod_b_1 c   value_1 -0.60" in txt
   and "retention 0.70   curve_direction concept_at_end   curve_shape ease   curve_value 1.00   scramble_seed -1" in txt
   and "curve_direction concept_at_start   curve_shape tanh   curve_value 0.50" in txt
   and "prompt_hint: identity: ginger woman" in txt, txt)
txt2 = ra.comfy_readout(rows, retention=1.0, frame_curve=("constant", "linear", 1.0), scramble_seed=3,
                        step_curve=("constant", "linear", 1.0), step_on=True)
ck("flat step curve reads as not connected", "Step Curve: not connected" in txt2)
ck("prompt hint: only mods with a description", ra.prompt_hint([ma, mb]) == "identity: ginger woman")

bz, tags, note = ra.bake(za, {"tags": ["4 img"]}, strength=0.8, retention=0.7, curve=("concept_at_end", "ease", 1.0))
ck("bake = ref_block at strength x retention with the frame curve, tags appended",
   torch.equal(bz, ra.ref_block(za, 0.56, ("concept_at_end", "ease", 1.0))) and tags == ["4 img", "studio: strength 0.56, curve concept_at_end/ease/1.00"]
   and "copies" in note)
try:
    ra.bake(za, {}, strength=0.0, retention=1.0, curve=None)
    ck("bake refuses strength 0", False)
except ValueError:
    ck("bake refuses strength 0", True)

# ── files: scan + graph preset ──────────────────────────────────────────────────────────────
import tempfile  # noqa: E402
from fizgig.minimax.refmod import save_refmod  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    save_refmod(os.path.join(td, "vid"), _mod(3), name="vid", mode="encode", pool="3x16x8", optimize_steps=200,
                description="a face", concept_type="identity")
    save_refmod(os.path.join(td, "img"), _mod(1), name="img", mode="encode", pool="1x16x8", optimize_steps=0)
    with open(os.path.join(td, "not_a_mod.safetensors"), "wb") as f:
        f.write(b"\x00" * 16)
    cache = {}
    found = ra.scan_refmods(td, cache)
    ck("scan finds the two mods, skips the junk file, sorted by name",
       [m["name"] for m in found] == ["img", "vid"] and found[1]["tokens"] == 96 and found[1]["kind"] == "video"
       and found[0]["kind"] == "image", str([m.get("name") for m in found]))
    ck("scan caches by (path,size,mtime)", len(cache) == 2 and ra.scan_refmods(td, cache) == found)
    ck("describe line", ra.describe_meta(found[1]) == 'video · 3×16×8 · 96 tokens · optimised 200 · "a face"',
       ra.describe_meta(found[1]))
    ck("plain encode wording for 0 steps", "plain encode" in ra.describe_meta(found[0]))
    from PIL import Image, PngImagePlugin
    info = PngImagePlugin.PngInfo()
    info.add_text("graph", '{"direction": "concept_at_middle", "shape": "sigmoid", "value": 0.8}')
    gp = os.path.join(td, "mid.png")
    Image.new("RGB", (8, 8)).save(gp, pnginfo=info)
    ck("reads the pack's PNG graph preset", ra.read_graph_preset_png(gp) == ("concept_at_middle", "sigmoid", 0.8))
    Image.new("RGB", (8, 8)).save(os.path.join(td, "plain.png"))
    ck("a plain PNG is not a preset", ra.read_graph_preset_png(os.path.join(td, "plain.png")) is None)
    bp = os.path.join(td, "baked")
    out = ra.save_baked(bp, bz, found[1], tags)
    from fizgig.minimax.refmod import load_refmod
    lz, lm = load_refmod(out)
    ck("baked file loads: latent == bake output (fp16), tags carried, kind video",
       torch.allclose(lz, bz.half().float()) and lm["tags"][-1].startswith("studio:") and lm["kind"] == "video")

print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
