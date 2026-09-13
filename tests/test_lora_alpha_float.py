"""A LoRA module built from a file's tensor alpha must carry a plain Python float scale.

13 Sep 2026: Krea 2 + torch.compile + a context LoRA crashed at the first training step with
GuardOnDataDependentSymNode. The file's `.alpha` tensor was turned into a NumPy scalar in
LoRAModule.__init__, `self.scale` inherited that, and the inference epilogue's
`torch.add(..., alpha=float(m) * float(self.scale))` then traced `float(scale)` as a
data-dependent value under compile. A NumPy scalar and a Python float behave identically
outside compile, so nothing else caught it. This pins the type for every construction path.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.networks.lora import LoRAModule, LoRAInfModule  # noqa: E402

fails = 0


def ck(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


lin = torch.nn.Linear(16, 16)
cases = {
    "python int alpha": 8,
    "python float alpha": 8.0,
    "fp32 tensor alpha (file)": torch.tensor(8.0),
    "bf16 tensor alpha (file)": torch.tensor(8.0, dtype=torch.bfloat16),
    "None alpha (defaults to rank)": None,
    "zero alpha (defaults to rank)": 0,
}
for cls in (LoRAModule, LoRAInfModule):
    for name, alpha in cases.items():
        m = cls("lora_unet_t", torch.nn.Linear(16, 16), 1.0, 8, alpha)
        ck(f"{cls.__name__}: {name} -> scale is a Python float",
           type(m.scale) is float, f"got {type(m.scale).__name__}")
        ck(f"{cls.__name__}: {name} -> scale value 1.0", abs(m.scale - 1.0) < 1e-9, repr(m.scale))
        ck(f"{cls.__name__}: {name} -> alpha buffer is a tensor of 8",
           isinstance(m.alpha, torch.Tensor) and float(m.alpha) == 8.0, repr(m.alpha))

# a non-unit scale survives as a float too
m = LoRAInfModule("lora_unet_t", torch.nn.Linear(16, 16), 1.0, 8, torch.tensor(4.0))
ck("tensor alpha 4 over rank 8 -> scale 0.5 float", type(m.scale) is float and m.scale == 0.5, repr(m.scale))

# the inference epilogue: add(alpha=) path with a float multiplier and float scale
m.apply_to()
x = torch.randn(3, 16)
with torch.no_grad():
    m.lora_up.weight.fill_(0.01)
    m.lora_down.weight.fill_(0.01)
ref = m.org_forward(x) + m.lora_up(m.lora_down(x)) * 1.0 * 0.5
ck("inference forward matches multiplier x scale maths", torch.allclose(m.forward(x), ref, atol=1e-6))

print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
