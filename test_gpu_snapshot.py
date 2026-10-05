"""Hardware-free checks for isolated Windows ROCm discovery."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from fizgig.utils import capabilities as caps


class SnapshotTests(unittest.TestCase):
    def test_default_detection_preserves_unaffected_backends(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True,
                get_device_properties=lambda i: SimpleNamespace(name="GPU", total_memory=16*2**30),
                get_device_capability=lambda i: (8, 9),
                mem_get_info=lambda i: (14*2**30, 16*2**30)),
            float8_e4m3fn="fp8", int8="int8",
            backends=SimpleNamespace(cuda=SimpleNamespace(cudnn_sdp_enabled=True)))
        for platform, rocm in (("nt", False), ("posix", False), ("posix", True)):
            with self.subTest(platform=platform, rocm=rocm):
                caps.detect.cache_clear()
                with patch.dict(sys.modules, {"torch": torch, "flash_attn": SimpleNamespace(),
                                              "bitsandbytes": SimpleNamespace()}), \
                     patch.object(caps.os, "name", platform), \
                     patch.object(caps, "is_rocm", return_value=rocm), \
                     patch.object(caps, "_probe_scaled_mm", return_value=True) as scaled, \
                     patch.object(caps, "_probe_int_mm", return_value=True) as integer:
                    result = caps.detect()
                    self.assertEqual(scaled.call_count, 1 if rocm else 2)
                    integer.assert_called_once_with()
                    self.assertEqual(result.fp8_matmul, not rocm)
                    self.assertTrue(result.int8_matmul_train)
                    self.assertTrue(result.flash_attn)
                    self.assertTrue(result.bitsandbytes)
                    self.assertEqual(result.vram_free_gb, 14)
                    self.assertEqual(result.cudnn_attention, not rocm)
        caps.detect.cache_clear()

    def test_helper_passes_explicit_probe_policy_and_serializes_caps(self):
        script = Path(__file__).resolve().parent / "src/fizgig/scripts/gpu_snapshot.py"
        spec = importlib.util.spec_from_file_location("gpu_snapshot", script)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        from io import StringIO
        for arguments, expected in (([], False), (["--probe-kernels"], True)):
            output = StringIO()
            with patch.object(sys, "argv", [str(script)] + arguments), \
                 patch.object(helper, "detect", return_value=caps.Capabilities(has_cuda=True)) as detect, \
                 patch.object(sys, "stdout", output):
                helper.main()
            detect.assert_called_once_with(probe_kernels=expected)
            self.assertTrue(caps.Capabilities(**json.loads(output.getvalue())).has_cuda)


if __name__ == "__main__":
    unittest.main()
