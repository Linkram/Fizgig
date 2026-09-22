"""CPU-only behavior tests; GPU throughput is measured by benchmarks/probe_krea_full.py."""
import os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import torch
from safetensors.torch import save_file
from fizgig.modules import rdna2_linear as fast
from fizgig.modules.nf4 import _RDNA2NF4Linear
from fizgig.krea2.model import SingleMMDiTConfig,SingleStreamDiT
from fizgig.krea2.nf4_loader import load_nf4_streamed
from fizgig.utils.capabilities import Capabilities,recommend_krea2_strategy

class RDNA2Tests(unittest.TestCase):
    def test_architecture_detection_covers_rdna2_family(self):
        from fizgig.utils import gpu_backend
        for arch, expected in (('gfx1030',True), ('gfx1031',True),
                               ('gfx1032',True), ('gfx1035:sramecc-:xnack-',True),
                               ('gfx1100',False), ('gfx1200',False), ('gfx900',False)):
            with self.subTest(arch=arch):
                fast._is_rdna2.cache_clear()
                with patch.object(gpu_backend,'is_rocm',return_value=True),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(gcnArchName=arch)):
                    self.assertEqual(fast._is_rdna2(0),expected)
        fast._is_rdna2.cache_clear()
        with patch.object(gpu_backend,'is_rocm',return_value=False),patch.object(torch.cuda,'get_device_properties') as props:
            self.assertFalse(fast._is_rdna2(0))
            props.assert_not_called()
        fast._is_rdna2.cache_clear()

    def test_frozen_linear_output_and_input_gradient_match_fp32_reference(self):
        torch.manual_seed(7)
        x=torch.randn(2,3,32,dtype=torch.bfloat16,requires_grad=True)
        w=torch.randn(48,32,dtype=torch.bfloat16)
        g=torch.randn(2,3,48,dtype=torch.bfloat16)
        with patch.dict(os.environ,{'FIZGIG_RDNA2_LINEAR':'fp32'}),torch.autocast('cpu',dtype=torch.bfloat16):
            y=fast._FrozenLinear.apply(x,w)
            y.backward(g)
        self.assertTrue(torch.equal(y,(x.float()@w.float().t()).bfloat16()))
        self.assertTrue(torch.equal(x.grad,(g.float()@w.float()).bfloat16()))

    def test_nf4_backward_rebuilds_weight_and_keeps_packed_storage(self):
        x=torch.randn(2,32,dtype=torch.bfloat16,requires_grad=True)
        w=torch.randn(48,32,dtype=torch.bfloat16);packed=torch.zeros(1,dtype=torch.uint8)
        state=object();calls=[]
        def dequant(p,s):
            self.assertIs(p,packed);self.assertIs(s,state);calls.append(1);return w
        with patch.dict(sys.modules,{'bitsandbytes.functional':SimpleNamespace(dequantize_nf4=dequant)}),patch.dict(os.environ,{'FIZGIG_RDNA2_LINEAR':'fp32'}):
            y=_RDNA2NF4Linear.apply(x,packed,state)
            self.assertIs(y.grad_fn.packed,packed)
            y.sum().backward()
        self.assertEqual(len(calls),2)
        self.assertTrue(torch.equal(x.grad,(torch.ones(2,48)@w.float()).bfloat16()))

    def test_cpu_and_disabled_paths_remain_original(self):
        x=torch.randn(2,32,dtype=torch.bfloat16,requires_grad=True)
        w=torch.randn(48,32,dtype=torch.bfloat16,requires_grad=True)
        with patch.dict(os.environ,{'FIZGIG_RDNA2_LINEAR':'fp32'}):
            self.assertFalse(fast.enabled(x))
            fast.frozen_linear(x,w).sum().backward()
        self.assertIsNotNone(w.grad)
        with patch.dict(os.environ,{'FIZGIG_RDNA2_LINEAR':'0'}),patch.object(fast,'_is_rdna2') as probe:
            self.assertFalse(fast.enabled(x));probe.assert_not_called()

    def test_auto_prefers_nf4_for_detected_rdna2_without_launcher(self):
        caps=Capabilities(has_cuda=True,is_rocm=True,gcn_arch='gfx1030',bitsandbytes=True,int8_matmul_train=True,vram_gb=16,vram_free_gb=16)
        with patch.dict(os.environ,{},clear=True):
            plan=recommend_krea2_strategy(caps=caps,vram_gb=16,mp=0.5,rank=16)
            self.assertTrue(plan.quant_4bit)
            explicit=recommend_krea2_strategy(caps=caps,vram_gb=16,mp=0.5,rank=16,force_quant='int8')
            self.assertFalse(explicit.quant_4bit)
            caps.gcn_arch='gfx1100'
            unchanged=recommend_krea2_strategy(caps=caps,vram_gb=24,mp=0.25,rank=16)
            self.assertFalse(unchanged.quant_4bit)
            self.assertNotIn('RDNA2',unchanged.reason)

    def test_hardware_dispatch_and_streaming_without_environment_flags(self):
        x=SimpleNamespace(device=torch.device('cuda:0'),dtype=torch.bfloat16)
        with patch.dict(os.environ,{},clear=True),patch.object(fast,'_is_rdna2',return_value=True):
            self.assertTrue(fast.enabled(x))
            self.assertTrue(fast.stream_nf4_enabled('cuda:0'))
            self.assertFalse(fast.stream_nf4_enabled('cpu'))
        with patch.dict(os.environ,{},clear=True),patch.object(fast,'_is_rdna2',return_value=False):
            self.assertFalse(fast.enabled(x))
            self.assertFalse(fast.stream_nf4_enabled('cuda:0'))
        with patch.dict(os.environ,{'FIZGIG_RDNA2_LINEAR':'0','FIZGIG_STREAM_NF4':'0'}),patch.object(fast,'_is_rdna2') as probe:
            self.assertFalse(fast.enabled(x))
            self.assertFalse(fast.stream_nf4_enabled('cuda:0'))
            probe.assert_not_called()

    def test_rocm_auto_compile_never_detects_or_probes_gpu(self):
        from fizgig.utils import capabilities
        with patch.object(capabilities,'is_rocm',return_value=True),patch.object(capabilities,'detect',side_effect=AssertionError('must not probe')):
            result,reason=capabilities.should_compile(1000,True,'',0)
        self.assertFalse(result)
        self.assertIn('ROCm',reason)

    def test_streaming_loader_scope_values_and_validation(self):
        config=SingleMMDiTConfig(features=128,tdim=32,txtdim=128,heads=1,multiplier=1,layers=1,patch=2,channels=16,txtheads=1,txtlayers=2)
        original=SingleStreamDiT(config).bfloat16()
        state=original.state_dict();seen=[]
        def quantize(w,compress_statistics):
            self.assertFalse(compress_statistics);seen.append(w.clone());return w.clone(),object()
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'model.safetensors');save_file(state,path)
            with patch.dict(sys.modules,{'bitsandbytes.functional':SimpleNamespace(quantize_nf4=quantize)}):
                loaded=load_nf4_streamed(path,device='cpu',config=config)
            self.assertEqual(len(seen),8)
            for name,module in loaded.named_modules():
                if getattr(module,'_is_nf4',False):
                    self.assertTrue(name.startswith('blocks.'))
                    self.assertTrue(torch.equal(module._nf4_packed,state[name+'.weight']))
                else:
                    for key,value in module.named_parameters(recurse=False):
                        self.assertTrue(torch.equal(value,state[name+'.'+key] if name else state[key]))
            state.pop(next(iter(state)));save_file(state,path)
            with patch.dict(sys.modules,{'bitsandbytes.functional':SimpleNamespace(quantize_nf4=quantize)}):
                with self.assertRaisesRegex(ValueError,'complete'):load_nf4_streamed(path,device='cpu',config=config)

if __name__=='__main__':unittest.main()
