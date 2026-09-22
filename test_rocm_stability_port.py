"""Hardware-free regression checks for the Windows ROCm stability port."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent

def load(relative):
    spec = importlib.util.spec_from_file_location('stability_module', ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.monitor = load('src/fizgig/utils/vram_monitor.py')

    def test_idle_monitor_does_not_initialize_gpu(self):
        cuda = Mock()
        cuda.is_initialized.return_value = False
        with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}):
            reader = self.monitor._TypeperfVramReader()
            with patch.object(reader, '_ensure_started') as start:
                self.assertIsNone(reader.read())
                start.assert_not_called()
            reader.close()
        cuda.is_available.assert_not_called()
        cuda.get_device_properties.assert_not_called()
        cuda.mem_get_info.assert_not_called()

    def test_missing_torch_is_not_imported(self):
        with patch.dict(sys.modules, {'torch': None}):
            self.assertIsNone(self.monitor._total_vram_from_torch())

    def test_windows_missing_counters_never_fall_back_to_hip(self):
        with patch.object(self.monitor.os, 'name', 'nt'), patch.object(self.monitor, '_read_vram_windows_typeperf', return_value=None), patch.object(self.monitor, '_read_vram_torch_fallback') as hip:
            self.assertIsNone(self.monitor.read_amd_gpu_vram())
            hip.assert_not_called()

    def test_reader_returns_cached_sample_without_one_shot_process(self):
        reader = self.monitor._TypeperfVramReader()
        reader._total = 100
        with patch.object(reader, '_ensure_started'), patch.object(self.monitor.subprocess, 'run') as run:
            self.assertIsNone(reader.read())
            reader._latest_used = 20
            self.assertEqual(reader.read(), (20, 100))
            reader._failed = True
            self.assertIsNone(reader.read())
            run.assert_not_called()
        reader._proc = Mock()
        reader.close()
        reader._proc.kill.assert_called_once()

    def test_backend_detection_does_not_probe_device(self):
        cuda = Mock(side_effect=AssertionError('device probe'))
        fake = SimpleNamespace(cuda=cuda, version=SimpleNamespace(rocm='7.15', hip=None))
        with patch.dict(sys.modules, {'torch': fake}):
            self.assertTrue(load('src/fizgig/utils/gpu_backend.py').is_rocm())
        cuda.is_available.assert_not_called()

    def test_preview_cleanup_after_decode_and_parking_failures(self):
        # Execute the actual decode/cleanup block with fake models, without importing torch.
        tree = ast.parse((ROOT / 'src/fizgig/krea2/sampling.py').read_text())
        sample = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sample')
        start = next(i for i,n in enumerate(sample.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='parked_for_decode' for t in n.targets))
        code = compile(ast.Module(body=sample.body[start:start+2], type_ignores=[]), '<decode cleanup>', 'exec')
        for failure in ('decode', 'park', 'cpu'):
            with self.subTest(failure=failure):
                events=[]
                def move(device):
                    events.append(device)
                    if device=='cpu' and failure=='cpu': raise RuntimeError('cpu')
                    return ae
                def park():
                    events.append('park')
                    if failure=='park': raise RuntimeError('park')
                ae=SimpleNamespace(to=move, decode_to_pixels=Mock(side_effect=RuntimeError('decode')))
                env={'ae':ae,'img':SimpleNamespace(device='cuda',to=lambda dtype:None),'torch':SimpleNamespace(bfloat16='bf16'),'before_decode':park,'after_decode':lambda:events.append('restore')}
                with self.assertRaises(RuntimeError): exec(code,env)
                self.assertEqual(events[-2:],['cpu','restore'])

    def test_budget_reclaims_cache_and_keeps_headroom(self):
        tree=ast.parse((ROOT/'src/fizgig/utils/device.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='auto_available_vram_gib')
        gib=1024**3
        cuda=SimpleNamespace(current_device=lambda:0,memory_reserved=lambda i:4*gib,memory_allocated=lambda i:2*gib,get_device_properties=lambda i:SimpleNamespace(total_memory=16*gib))
        env={'torch':SimpleNamespace(cuda=cuda,device=lambda *args:0),'plannable_free_vram':lambda d:8*gib/1e9}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<budget>','exec'),env)
        self.assertEqual(env['auto_available_vram_gib'](),9)
        env['plannable_free_vram']=lambda d:16*gib/1e9
        self.assertEqual(env['auto_available_vram_gib'](),15)

if __name__ == '__main__':
    unittest.main()
