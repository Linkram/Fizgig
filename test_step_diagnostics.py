import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import torch
from fizgig.training.step_diagnostics import StepDiagnostics


class StepDiagnosticsTests(unittest.TestCase):
    def test_disabled_does_not_touch_gpu_or_batch(self):
        with patch.dict(os.environ, {"FIZGIG_STEP_DIAGNOSTICS": "0"}):
            profiler = StepDiagnostics("cuda", default_steps=3)
        with patch.object(torch.cuda, "synchronize") as sync:
            profiler.begin(1, {})
            profiler.mark("forward")
            sync.assert_not_called()

    def test_phase_timing_synchronizes_and_stops_at_limit(self):
        with patch.dict(os.environ, {"FIZGIG_STEP_DIAGNOSTICS": "1"}):
            profiler = StepDiagnostics("cuda")
        batch = {"latents": torch.empty(1,16,2,2), "attention_mask": torch.tensor([[True,True,False]])}
        with patch.object(torch.cuda, "synchronize") as sync, \
             patch.object(torch.cuda, "memory_allocated", return_value=2**30), \
             patch.object(torch.cuda, "memory_reserved", return_value=2*2**30), \
             patch("fizgig.training.step_diagnostics.time.perf_counter", side_effect=[10,13,14]), \
             self.assertLogs("fizgig.training.step_diagnostics", level="INFO") as logs:
            profiler.begin(1, batch)
            profiler.mark("forward")
            profiler.begin(2, {})
            profiler.mark("forward")
        self.assertEqual(sync.call_count, 2)
        self.assertIn("valid_text_tokens=[2]", logs.output[0])
        self.assertIn("forward=3.000s", logs.output[1])
        self.assertIn("reserved=2.00 GiB", logs.output[1])


if __name__ == "__main__":
    unittest.main()
