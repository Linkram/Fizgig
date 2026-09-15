"""The training adapter under H3 fine-tune rides as FORWARD HOOKS (attach_frozen_lora_hooks).

Why hooks: LoRAModule.apply_to sets an instance `forward` on each Linear that captures the OLD
bound forward, and the component rotator swaps `lin.__class__` (Linear4bit <-> nn.Linear) on
the same instance every window — the captured forward then runs the wrong class's code. A
forward hook is stored on the instance and runs after whatever forward is live.

CPU-only, a stand-in DiT with H3-shaped module names. Pins: bit-identical to the apply_to
path; survives the rotator's class + weight swap; enabled=False is a passthrough (parked too);
grads reach the swapped-in weight, never the adapter; net.to() leaves the base alone (the
org_module de-registration); the Turbo's apply_to/pop bracket composes; the planner counts
the adapter under FT and still excludes the Context LoRA there; the trainer's load block routes
by rotator; the CLI help no longer says "not available".
"""
import copy
import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.minimax import trainer as T  # noqa: E402

fails = 0


def ck(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        q = self.qkv(x)[..., :x.shape[-1]]
        return self.out(q)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = Attn(d)
        self.fc1 = nn.Linear(d, 2 * d, bias=False)
        self.fc2 = nn.Linear(2 * d, d, bias=False)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.fc2(torch.relu(self.fc1(x)))


class TinyDiT(nn.Module):
    def __init__(self, d=16, n=3):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class OtherLinear(nn.Linear):
    """Stands in for bnb's Linear4bit: a subclass whose forward differs (scaled by 2)."""
    def forward(self, x):
        return 2.0 * nn.functional.linear(x, self.weight, self.bias)


torch.manual_seed(0)
D = 16
dit = TinyDiT(D)
x = torch.randn(2, 5, D)

# a rank-2 kohya adapter over every Linear in the stand-in
sd = {}
for name, m in dit.named_modules():
    if isinstance(m, nn.Linear):
        k = f"lora_unet_{name.replace('.', '_')}"
        sd[f"{k}.lora_down.weight"] = torch.randn(2, m.in_features) * 0.3
        sd[f"{k}.lora_up.weight"] = torch.randn(m.out_features, 2) * 0.3
        sd[f"{k}.alpha"] = torch.tensor(2.0)
n_lin = len(sd) // 3

with tempfile.TemporaryDirectory() as td:
    from safetensors.torch import save_file
    path = os.path.join(td, "adapter.safetensors")
    save_file(sd, path, metadata={"ss_architecture": "minimax"})

    # reference: the apply_to path (what LoRA mode does) on a deep copy
    ref = copy.deepcopy(dit)
    ref_net, _pairs = T.load_context_lora(ref, path, 1.0, "cpu", torch.float32, tag="t", label="t")
    with torch.no_grad():
        y_ref = ref(x)
        y_base = dit(x)

    hooks = T.attach_frozen_lora_hooks(dit, path, 1.0, "cpu", torch.float32, tag="t", label="t")
    ck(f"every Linear hooked ({n_lin})", hooks.n_modules == n_lin, hooks.n_modules)
    with torch.no_grad():
        y_hook = dit(x)
    ck("hooked output == apply_to output (bit-identical)", torch.equal(y_hook, y_ref),
       f"max diff {(y_hook - y_ref).abs().max().item():.3e}")
    ck("…and differs from the plain base", not torch.allclose(y_hook, y_base))
    ck("adapter params frozen", all(not p.requires_grad for p in hooks.net.parameters()))
    ck("net.to() leaves the base out (org_module de-registered)",
       all("org_module" not in m._modules for m in hooks.net.unet_loras))
    base_ptr = dit.blocks[0].attn.qkv.weight.data_ptr()
    hooks.net.to(torch.float64)
    ck("moving the adapter net does not touch the base Linear",
       dit.blocks[0].attn.qkv.weight.dtype == torch.float32 and dit.blocks[0].attn.qkv.weight.data_ptr() == base_ptr)
    hooks.net.to(torch.float32)

    # the rotator's move: class swap + a fresh Parameter on the SAME instance
    lin = dit.blocks[1].attn.qkv
    w_master = lin.weight.detach().clone()
    orig_cls = type(lin)
    lin.__class__ = OtherLinear
    lin.weight = nn.Parameter(w_master.clone(), requires_grad=True)
    with torch.no_grad():
        y_swapped = dit(x)
    # expected: the new class's forward (2x) on that Linear, plus the adapter's delta, elsewhere unchanged
    m1 = next(m for m in hooks.net.unet_loras if m.lora_name == "lora_unet_blocks_1_attn_qkv")
    hooks.set_enabled(False)
    with torch.no_grad():
        y_swapped_noad = dit(x)
    hooks.set_enabled(True)
    ck("after the class + weight swap the hook still adds the delta (output changes with it)",
       not torch.equal(y_swapped, y_swapped_noad))
    # precise: hook on the swapped module = OtherLinear forward + delta
    h_in = torch.randn(3, D)
    with torch.no_grad():
        got = lin(h_in)
        want = 2.0 * nn.functional.linear(h_in, lin.weight) + m1.compute_delta(h_in)
    ck("swapped module: new class forward + adapter delta, exactly", torch.equal(got, want))
    lin.__class__ = orig_cls
    with torch.no_grad():
        back = lin(h_in)
        want_back = nn.functional.linear(h_in, lin.weight) + m1.compute_delta(h_in)
    ck("swapped back: original forward + adapter delta", torch.equal(back, want_back))

    # gradients: to the swapped-in weight and the input, never to the adapter
    lin.__class__ = OtherLinear
    xg = x.clone().requires_grad_(True)
    dit(xg).sum().backward()
    ck("grad reaches the window's fresh Parameter", lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0)
    ck("grad reaches the input", xg.grad is not None and xg.grad.abs().sum() > 0)
    ck("no grad on the adapter", all(p.grad is None for p in hooks.net.parameters()))
    lin.__class__ = orig_cls

    # enabled False = passthrough, parked too (the preview bracket)
    hooks.set_enabled(False)
    with torch.no_grad():
        y_off = dit(x)
    ck("enabled=False is a passthrough (== plain base, module-by-module)",
       torch.allclose(y_off, TinyDiT.forward(dit, x)) and not torch.allclose(y_off, y_hook))
    moved = hooks.park()
    with torch.no_grad():
        y_off2 = dit(x)
    ck("parked + disabled still forwards (no device fault, same output)", torch.equal(y_off2, y_off))
    hooks.restore("cpu")
    hooks.set_enabled(True)
    with torch.no_grad():
        y_on = dit(x)
    ck("restored + enabled = back to the hooked output", torch.equal(y_on, y_hook))

    # the Turbo bracket: an apply_to-style instance forward on top, then popped
    turbo_net, _ = T.load_preview_turbo(dit, path, 0.5, tag="turbo")   # applies via apply_to, disabled
    for m in turbo_net.unet_loras:
        m.enabled = True
    with torch.no_grad():
        y_both = dit(x)
    for _l in turbo_net.unet_loras:
        _m = getattr(getattr(_l, "org_forward", None), "__self__", None)
        if _m is not None:
            _m.__dict__.pop("forward", None)
    with torch.no_grad():
        y_after = dit(x)
    ck("a wrapped-forward LoRA on top composes (output moves) and pops clean (back to hooked)",
       not torch.equal(y_both, y_hook) and torch.equal(y_after, y_hook))

    hooks.remove()
    with torch.no_grad():
        y_removed = dit(x)
    ck("remove() detaches every hook", hooks.n_modules == 0 and torch.equal(y_removed, y_base))

    # the planner counts the adapter under FT, and still not a Context LoRA
    P = 1_000_000
    t0, fr0, em0 = T.plan_adapter_gb(P, "adamw", ft_rotation=4)
    t1, fr1, em1 = T.plan_adapter_gb(P, "adamw", training_adapter_path=path, ft_rotation=4)
    t2, fr2, em2 = T.plan_adapter_gb(P, "adamw", training_adapter_path=path, context_lora_path=path,
                                     ema_decay=0.98, ft_rotation=4)
    ck("planner: adapter counted under FT", fr1 > 0 and t1 == t0 + fr1, (t0, fr1, t1))
    ck("planner: context LoRA and EMA still excluded under FT", fr2 == fr1 and em2 == 0.0, (fr2, em2))

# the trainer routes by rotator; the CLI no longer refuses
src = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "minimax", "trainer.py"), encoding="utf-8").read()
ck("load block: hooks under rotation, context still refused there",
   "adapter_hooks = attach_frozen_lora_hooks(dit, training_adapter_path, 1.0," in src
   and 'raise RuntimeError("Context LoRA is not available with fine-tuning on MiniMax H3 "' in src
   and "The training adapter\" if training_adapter_path else" not in src)
ck("preview brackets park/restore the hooks", "adapter_hooks.set_enabled(False)" in src
   and "adapter_hooks.restore(device)" in src and "adapter_hooks.set_enabled(True)" in src)
cli = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "scripts", "minimax_train.py"), encoding="utf-8").read()
_ad_help = cli[cli.index("--training_adapter_path"):cli.index("--tread_ratio")]
ck("CLI help: the adapter is available under --finetune_rotation (the Context LoRA still isn't)",
   "Not available with --finetune_rotation" not in _ad_help and "rides as forward hooks" in _ad_help)

print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
