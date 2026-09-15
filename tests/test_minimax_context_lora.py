"""MiniMax H3 Context LoRA — a frozen, active LoRA under the trainable one.

Pins, on a tiny bf16 H3 shape through the REAL loader:
  1. an AI-Toolkit-format file (diffusion_model.blocks.N....lora_A/B) loads as context and
     the forward equals base-with-the-LoRA-merged-by-hand (exactness, not "runs");
  2. the trainable network stacks on top: at zero init the output is unchanged, and a
     backward reaches every trainable module and NO context tensor;
  3. the AdaLN injection COMPOSES (list of updates per module, effects add); the training
     adapter stacks under the context as a second frozen layer; previews drop the ADAPTER
     and keep the CONTEXT (per-layer contract), and the training stack comes back after;
  4. ss_context_lora / ss_context_lora_strength ride in the run provenance (static pin);
  5. the GUI emits --context_lora_path/--context_lora_strength for MiniMax and refuses the
     fine-tune combination at validation.

Run: venv/Scripts/python.exe tests/test_minimax_context_lora.py   (CUDA optional)
"""
import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
os.environ["FIZGIG_NO_PERSIST"] = "1"

import fizgig.minimax.loader as L                                            # noqa: E402
from fizgig.minimax.model import MiniMaxH3Config, MiniMaxH3DiT               # noqa: E402
from fizgig.minimax import trainer as T                                      # noqa: E402
from fizgig.networks.lora import create_network                              # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.bfloat16
torch.manual_seed(0)

# --- tiny bf16 (non-pruned) H3 shape through the real loader, no quantization ------------
tiny = MiniMaxH3Config(hidden_size=64, num_layers=4, token_refiner_num_layers=1,
                       num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=64,
                       latents_dim=24, audio_latents_dim=6, patch_size=(1, 2, 2), text_dim=24,
                       time_embed_dim=8, rope_inv_freq_len=2)
shape = MiniMaxH3DiT(tiny)
sd = {n: (torch.randn(p.shape) * 0.02).to(torch.bfloat16) for n, p in shape.named_parameters()}
sd.update({n: torch.randn(b.shape) * 0.02 for n, b in shape.named_buffers()})
# Random 0.02 weights leave every AdaLN gate ~1e-3, so the blocks barely reach the output
# (zeroing all of them moved it 1.2e-4 — under bf16 resolution). Bias the modulation to 1 so
# block contributions are visible and the exactness pin has something to measure.
for k in list(sd):
    if "adaln" in k and k.endswith(".bias"):
        sd[k] = torch.ones_like(sd[k])
tmp = tempfile.mkdtemp(prefix="mmh3_ctx_")
base_path = os.path.join(tmp, "tiny_bf16.safetensors")
save_file({k: v.contiguous() for k, v in sd.items()}, base_path)
_real_cfg = L.config_from_checkpoint
L.config_from_checkpoint = lambda keys, table_shape=None: tiny


def build():
    dit = L.load_minimax_h3_dit(base_path, device=str(DEV), compute_dtype=DT, quantize=False)
    dit.requires_grad_(False)
    return dit


def inputs(seed=5):
    g = torch.Generator().manual_seed(seed)
    lat = torch.randn(1, 24, 1, 8, 8, generator=g).to(DEV, DT)
    txt = torch.randn(1, 7, 24, generator=g).to(DEV, DT)
    return lat, torch.tensor([0.4], device=DEV), txt


# --- an AI-Toolkit-format context LoRA over attn + mlp of every block, rank 4 --------------
RANK, STRENGTH = 4, 0.7
ctx_sd, ctx_dense = {}, {}     # dense[module dotted name] = strength * scale * B @ A
with torch.no_grad():
    for name, m in shape.named_modules():
        if isinstance(m, torch.nn.Linear) and name.startswith("blocks.") and (
                ".attn." in name or ".mlp." in name):
            A = torch.randn(RANK, m.in_features) * 0.1
            B = torch.randn(m.out_features, RANK) * 0.1
            ctx_sd[f"diffusion_model.{name}.lora_A.weight"] = A.to(torch.bfloat16)
            ctx_sd[f"diffusion_model.{name}.lora_B.weight"] = B.to(torch.bfloat16)
            ctx_sd[f"diffusion_model.{name}.alpha"] = torch.tensor(float(RANK) / 2)   # ai-toolkit's alpha = rank/2
            ctx_dense[name] = STRENGTH * (0.5) * (B.to(torch.bfloat16).float() @ A.to(torch.bfloat16).float())
ctx_path = os.path.join(tmp, "aitk_context.safetensors")
save_file(ctx_sd, ctx_path, metadata={"software": '{"name": "ai-toolkit"}'})
ck("context file is ai-toolkit shaped", next(iter(ctx_sd)).startswith("diffusion_model.blocks."))

try:
    # 1. exactness: context-applied forward == hand-merged base --------------------------------
    dit_ref = build()
    with torch.no_grad():
        for name, m in dit_ref.named_modules():
            if name in ctx_dense:
                m.weight.add_(ctx_dense[name].to(m.weight.dtype).to(m.weight.device))
    lat, t, txt = inputs()
    with torch.no_grad():
        y_ref = dit_ref(lat, t, txt).float()
    del dit_ref

    dit = build()
    with torch.no_grad():
        y_base = dit(lat, t, txt).float()
    ctx_net, ctx_adaln = T.load_context_lora(dit, ctx_path, STRENGTH, DEV, DT)
    ck("context wraps every attn+mlp linear", len(ctx_net.unet_loras) == len(ctx_dense),
       f"{len(ctx_net.unet_loras)} vs {len(ctx_dense)}")
    ck("no adaln rows on the bf16 (non-pruned) shape", ctx_adaln == [])
    ck("context modules are enabled + frozen",
       all(m.enabled for m in ctx_net.unet_loras)
       and not any(p.requires_grad for p in ctx_net.parameters()))
    with torch.no_grad():
        y_ctx = dit(lat, t, txt).float()
    ck("context changes the forward", not torch.allclose(y_ctx, y_base))
    err = float((y_ctx - y_ref).abs().max()); scale = float(y_ref.abs().max())
    ck("context forward == base with the LoRA merged by hand", err <= 0.02 * scale + 1e-3,
       f"max |diff| {err:.3e} vs |y| {scale:.3e}")

    # 2. the trainable network stacks on top ---------------------------------------------------
    net = create_network(None, "lora_unet", 1.0, 4, 4, None, [], dit,
                         include_patterns=[r"blocks\.\d+\.attn\..*", r"blocks\.\d+\.mlp\..*"])
    net.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    net.requires_grad_(True)
    net.to(device=DEV, dtype=DT)
    with torch.no_grad():
        y_stack = dit(lat, t, txt).float()
    # bf16 tolerance: the same forward run twice differs by one ulp on CUDA (SDPA), so the
    # pin is "within rounding", not bitwise.
    ck("trainable LoRA at zero init leaves the context forward unchanged",
       torch.allclose(y_stack, y_ctx, atol=4e-3, rtol=1e-2),
       f"max |diff| {float((y_stack - y_ctx).abs().max()):.3e}")
    out = dit(lat, t, txt)
    out.float().pow(2).mean().backward()
    # At zero init only lora_up can receive gradient (lora_down's grad flows through the
    # zero lora_up) — one live tensor per module is the healthy signature.
    n_train = sum(1 for m in net.unet_loras
                  if m.lora_up.weight.grad is not None and m.lora_up.weight.grad.abs().sum() > 0)
    ck("backward reaches every trainable module", n_train == len(net.unet_loras),
       f"{n_train} of {len(net.unet_loras)} modules")
    ck("...and no context tensor takes part in the update",
       all(p.grad is None and not p.requires_grad for p in ctx_net.parameters()))

    # 3. AdaLN injection composes ---------------------------------------------------------------
    # Fake a pruned base's adaln: a module with .linear/.apply_silu/.modalities/.expand/.hidden,
    # a 3-row t table and a matching 3-row egrid; two (A, B) pairs on the SAME module.
    class _Ada(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 2 * 3 * 16, bias=True)
            self.apply_silu, self.modalities, self.expand, self.hidden = True, 2, 3, 16

        def forward(self, t_emb):
            x = self.linear(torch.nn.functional.silu(t_emb))
            return x.view(x.shape[0] * self.modalities, self.expand * self.hidden).chunk(self.expand, dim=-1)

    class _Dit:
        pruned_adaln = True
        adaln_t_table = torch.randn(3, 8)
    ada = _Ada().to(DEV)
    egrid = torch.randn(3, 32)
    pairs_c = [(ada, torch.randn(2, 32), torch.randn(2 * 3 * 16, 2))]
    pairs_t = [(ada, torch.randn(2, 32), torch.randn(2 * 3 * 16, 2))]
    temb = _Dit.adaln_t_table[1:2].to(DEV) + 0.01
    with torch.no_grad():
        plain = torch.cat(ada(temb), -1).float()
        T.turbo_adaln_patch(_Dit, pairs_c, DEV, torch.float32, egrid=egrid)
        only_c = torch.cat(ada(temb), -1).float()
        T.turbo_adaln_unpatch(pairs_c)
        T.turbo_adaln_patch(_Dit, pairs_t, DEV, torch.float32, egrid=egrid)
        only_t = torch.cat(ada(temb), -1).float()
        T.turbo_adaln_unpatch(pairs_t)
        n = T.turbo_adaln_patch(_Dit, pairs_c + pairs_t, DEV, torch.float32, egrid=egrid)
        both = torch.cat(ada(temb), -1).float()
        T.turbo_adaln_unpatch(pairs_t)          # what the preview-off path does...
        T.turbo_adaln_patch(_Dit, pairs_c, DEV, torch.float32, egrid=egrid)   # ...then re-patches context
        after = torch.cat(ada(temb), -1).float()
        T.turbo_adaln_unpatch(pairs_c)
        restored = torch.cat(ada(temb), -1).float()
    ck("each injection alone moves the output", not torch.allclose(only_c, plain)
       and not torch.allclose(only_t, plain))
    ck("context + turbo rows on one module patch ONE forward", n == 1)
    ck("...and their effects ADD", torch.allclose(both - plain, (only_c - plain) + (only_t - plain),
                                                  atol=1e-4, rtol=1e-4))
    ck("post-preview re-patch restores the context-only forward", torch.allclose(after, only_c))
    ck("unpatch returns the class forward", torch.allclose(restored, plain))

    # 3b. per-layer preview contract: the training adapter goes OFF for a preview, the user's
    # context LoRA stays ON. Exercised with the same primitives the trainer's closures use
    # (module .enabled flags + turbo_adaln_patch/unpatch), then pinned statically below.
    ad_sd, ad_dense = {}, {}
    with torch.no_grad():
        for name, m in shape.named_modules():
            if name in ctx_dense:
                A = torch.randn(RANK, m.in_features) * 0.1; B = torch.randn(m.out_features, RANK) * 0.1
                ad_sd[f"diffusion_model.{name}.lora_A.weight"] = A.to(torch.bfloat16)
                ad_sd[f"diffusion_model.{name}.lora_B.weight"] = B.to(torch.bfloat16)
                ad_dense[name] = (B.to(torch.bfloat16).float() @ A.to(torch.bfloat16).float())   # no alpha key -> scale 1
    ad_path = os.path.join(tmp, "training_adapter.safetensors")
    save_file(ad_sd, ad_path)
    dit2_ref = build()
    with torch.no_grad():
        for name, m in dit2_ref.named_modules():
            if name in ctx_dense:
                m.weight.add_((ctx_dense[name] + ad_dense[name]).to(m.weight.dtype).to(m.weight.device))
        y2_ref = dit2_ref(lat, t, txt).float()
    del dit2_ref
    dit2 = build()
    an, ap = T.load_context_lora(dit2, ad_path, 1.0, DEV, DT, tag="adapter", label="Training adapter")
    cn, cp = T.load_context_lora(dit2, ctx_path, STRENGTH, DEV, DT)
    with torch.no_grad():
        y2 = dit2(lat, t, txt).float()
    err2 = float((y2 - y2_ref).abs().max()); sc2 = float(y2_ref.abs().max())
    ck("adapter + context stacked == base with both merged by hand", err2 <= 0.02 * sc2 + 1e-3,
       f"max |diff| {err2:.3e} vs |y| {sc2:.3e}")
    with torch.no_grad():
        for m_ in an.unet_loras: m_.enabled = False          # what _frozen_for_preview does
        y2_prev = dit2(lat, t, txt).float()
        for m_ in an.unet_loras: m_.enabled = True           # what _frozen_for_training does
        y2_back = dit2(lat, t, txt).float()
    ck("preview view == base + context only (adapter off, context on)",
       torch.allclose(y2_prev, y_ctx, atol=4e-3, rtol=1e-2),
       f"max |diff| {float((y2_prev - y_ctx).abs().max()):.3e}")
    ck("...and the training view comes back", torch.allclose(y2_back, y2, atol=4e-3, rtol=1e-2))
    ck("neither frozen layer trains", not any(p.requires_grad for n_ in (an, cn) for p in n_.parameters()))
    src_t = open(os.path.join(REPO, "src", "fizgig", "minimax", "trainer.py"), encoding="utf-8").read()
    ck("preview-on drops the adapter and keeps the context (closure wired at the render site)",
       src_t.count("            _frozen_for_preview()") == 1
       and "turbo_adaln_patch(dit, context_adaln + turbo_adaln, device, dtype)" in src_t)
    ck("every preview-off site restores the training stack (normal + exception paths)",
       src_t.count("            _frozen_for_training()") == 2)
    del dit2, an, cn

    # 4. provenance keys (static) ----------------------------------------------------------------
    src = open(os.path.join(REPO, "src", "fizgig", "minimax", "trainer.py"), encoding="utf-8").read()
    ck("ss_context_lora + strength + ss_training_adapter in the run provenance",
       '"ss_context_lora":' in src and '"ss_context_lora_strength":' in src
       and '"ss_training_adapter":' in src)
    ck("fine-tune + context is refused in the trainer",
       "is not available with fine-tuning on MiniMax H3" in src)

    # 5. GUI: emits the flags for MiniMax, refuses the FT combination ---------------------------
    import tkinter as tk
    import lora_trainer_gui as G
    _real_save = G.save_prefs if hasattr(G, "save_prefs") else None
    root = tk.Tk(); root.withdraw()
    g = G.LoRATrainerGUI(root)
    try:
        g.architecture_var.set(next(k for k, v in G.ARCHITECTURES.items() if v.get("is_minimax")))
    except StopIteration:
        g.architecture_var.set("MiniMax H3")
    g.update_ui_for_architecture()
    g.entries["CONTEXT_LORA_PATH"].delete(0, tk.END); g.entries["CONTEXT_LORA_PATH"].insert(0, ctx_path)
    g.entries["CONTEXT_LORA_STRENGTH"].delete(0, tk.END); g.entries["CONTEXT_LORA_STRENGTH"].insert(0, "0.7")
    ck("Context LoRA row visible under MiniMax", g._contextlora_frame.winfo_manager() != "")
    g.settings["CONTEXT_LORA_PATH"] = ctx_path
    g.settings["CONTEXT_LORA_STRENGTH"] = "0.7"
    g.minimax_finetune_var.set(False)
    cmd = g._build_minimax_train_command()
    ck("MiniMax command carries --context_lora_path", "--context_lora_path" in cmd
       and cmd[cmd.index("--context_lora_path") + 1] == ctx_path)
    ck("...and --context_lora_strength", "--context_lora_strength" in cmd
       and cmd[cmd.index("--context_lora_strength") + 1] == "0.7")
    # training adapter tickbox: pref per base, flag emitted, refused under FT / without a file
    g.prefs_vars["minimax_training_adapter"].set(ad_path)
    g.prefs_vars["minimax_ref_training_adapter"].set(ctx_path)      # any file: the KEY is what's pinned
    g.entries["MINIMAX_TRAINING_ADAPTER"].set(True)
    g.settings["MINIMAX_TRAINING_ADAPTER"] = True
    g.settings["MINIMAX_TRAIN_BASE"] = "fl2va"; g.settings["MINIMAX_DISTILL"] = False
    cmd = g._build_minimax_train_command()
    ck("adapter tickbox emits --training_adapter_path (fl2va file)", "--training_adapter_path" in cmd
       and cmd[cmd.index("--training_adapter_path") + 1] == ad_path)
    g.settings["MINIMAX_TRAIN_BASE"] = "ref2va"
    cmd = g._build_minimax_train_command()
    ck("...and the ref2va file when the Training Base is ref2va",
       cmd[cmd.index("--training_adapter_path") + 1] == ctx_path)
    g.settings["MINIMAX_TRAIN_BASE"] = "fl2va"
    g.prefs_vars["minimax_training_adapter"].set("")
    shown0 = []
    _re0 = G.messagebox.showerror; G.messagebox.showerror = lambda title, msg, **k: shown0.append(msg)
    try:
        ok0 = g.validate_inputs()
    finally:
        G.messagebox.showerror = _re0
    ck("adapter ticked with no file in Preferences is refused at validation",
       ok0 is False and any("isn't set in Preferences" in m for m in shown0))
    g.prefs_vars["minimax_training_adapter"].set(ad_path)
    g.minimax_finetune_var.set(True)
    g._apply_minimax_ft_visibility()
    shown = []
    _real_err = G.messagebox.showerror
    G.messagebox.showerror = lambda title, msg, **k: shown.append(msg)
    try:
        ok = g.validate_inputs()
    finally:
        G.messagebox.showerror = _real_err
    errs = shown
    ck("fine-tune + context is refused at validation",
       ok is False and any("Context LoRA is not available with MiniMax H3 fine-tuning" in m for m in shown),
       (shown or ok))
    ck("under fine-tune the adapter tick stays visible, is not an error, and (ticked) emits the flag",
       not any("training adapter" in m.lower() for m in shown)
       and g._minimax_adapter_cb.winfo_manager() != ""
       and "--training_adapter_path" in g._build_minimax_train_command())
    ck("under fine-tune the hint says never in the checkpoint",
       "never in the checkpoint" in g._minimax_adapter_hint.cget("text"))
    # the FT recipe (pushed by the toggle handler on the way ON) ticks it ON (Peter, 15 Sep)
    g.entries["MINIMAX_TRAINING_ADAPTER"].set(False)
    g.settings["MINIMAX_TRAINING_ADAPTER"] = False
    g._on_minimax_ft_toggle()
    ck("the fine-tune recipe ticks the adapter ON and the command carries it",
       g.entries["MINIMAX_TRAINING_ADAPTER"].get() is True
       and "--training_adapter_path" in g._build_minimax_train_command())
    g.entries["MINIMAX_TRAINING_ADAPTER"].set(False)
    g.settings["MINIMAX_TRAINING_ADAPTER"] = False          # launch syncs entries -> settings
    ck("...unticked by hand it stays off under FT", "--training_adapter_path" not in g._build_minimax_train_command())
    g.entries["MINIMAX_TRAINING_ADAPTER"].set(True)
    g.settings["MINIMAX_TRAINING_ADAPTER"] = True
    g.minimax_finetune_var.set(False)
    g._apply_minimax_ft_visibility()
    ck("...and the LoRA-mode hint comes back when fine-tune is unticked",
       g._minimax_adapter_cb.winfo_manager() != ""
       and "Under fine-tune" not in g._minimax_adapter_hint.cget("text"))
    ck("adapter ships ON in every H3 preset",
       all(v.get("MINIMAX_TRAINING_ADAPTER") is True for v in G.MINIMAX_BUILT_IN_PRESETS.values())
       and len(G.MINIMAX_BUILT_IN_PRESETS) == 3)
    root.destroy()
finally:
    L.config_from_checkpoint = _real_cfg

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
