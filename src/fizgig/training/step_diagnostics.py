"""Bounded training-process diagnostics for separating GPU work from host overhead."""
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


class StepDiagnostics:
    """Time the first few steps; no background polling or GUI GPU context.

    GPU boundaries synchronize so asynchronous work is attributed to the correct
    phase. This deliberately applies only to a bounded startup sample.
    """
    def __init__(self, device, default_steps=0):
        self.device = torch.device(device)
        try:
            self.remaining = max(0, int(os.environ.get("FIZGIG_STEP_DIAGNOSTICS", default_steps)))
        except ValueError:
            self.remaining = default_steps
        self.active = False

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _memory(self):
        if self.device.type != "cuda":
            return ""
        gib = 1024 ** 3
        return (f", allocated={torch.cuda.memory_allocated(self.device)/gib:.2f} GiB"
                f", reserved={torch.cuda.memory_reserved(self.device)/gib:.2f} GiB")

    def begin(self, step, batch):
        self.active = self.remaining > 0
        if not self.active:
            return
        self.remaining -= 1
        self.step = step
        self._sync()
        valid = batch["attention_mask"].sum(dim=-1).tolist()
        logger.info("[step-profile] step=%s, latents=%s, valid_text_tokens=%s%s",
                    step, tuple(batch["latents"].shape), valid, self._memory())
        self.last = time.perf_counter()

    def mark(self, phase):
        if not self.active:
            return
        self._sync()
        now = time.perf_counter()
        logger.info("[step-profile] step=%s %s=%.3fs%s",
                    self.step, phase, now-self.last, self._memory())
        self.last = time.perf_counter()
