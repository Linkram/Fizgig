import os
from pathlib import Path
import sys
import unittest
import copy
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fizgig.modules.rdna2_experimental_attention import analytical_attention
from rejected_checkpoint import _Norm


class Experiments(unittest.TestCase):
    def test_default_gemm_still_matches_pr_fp32_path(self):
        from fizgig.modules.rdna2_linear import matmul
        a = torch.randn(9, 13, dtype=torch.bfloat16)
        b = torch.randn(13, 17, dtype=torch.bfloat16)
        with patch.dict(os.environ, {'FIZGIG_RDNA2_GEMM': ''}):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = matmul(a, b)
        torch.testing.assert_close(actual, (a.float() @ b.float()).bfloat16(), rtol=0, atol=0)

    def test_tiling_preserves_output_dtype_and_matrix_axes(self):
        from fizgig.modules.rdna2_linear import _split_output_fp16, _split_reduction_fp16
        torch.manual_seed(42)
        a = torch.randn(11, 17, dtype=torch.bfloat16) * .1
        b = torch.randn(23, 17, dtype=torch.bfloat16).t() * .1
        expected = (a.float() @ b.float()).bfloat16()
        for fn in (_split_output_fp16, _split_reduction_fp16):
            actual = fn(a, b, tile=8)
            self.assertEqual(actual.dtype, a.dtype)
            torch.testing.assert_close(actual.float(), expected.float(), rtol=.015, atol=.0003)

    def test_experiment_flags_do_not_install_on_other_architectures(self):
        from types import SimpleNamespace
        from fizgig.krea2.driver import Krea2Driver
        from fizgig.modules import rdna2_linear, rdna2_attention, sdpa
        flags = {'FIZGIG_RDNA2_HEADS': '2', 'FIZGIG_RDNA2_ATTN_IMPL': 'analytical',
                 'FIZGIG_RDNA2_GEMM': 'tiled-fp16'}
        with patch.dict(os.environ, flags), \
             patch.object(rdna2_linear, 'install_nf4_forward') as linear, \
             patch.object(rdna2_attention, 'install_attention') as attention, \
             patch.object(sdpa, '_SDPA_CTX', None):
            for arch in ('gfx1100', 'gfx1200', 'gfx90a', ''):
                with patch.object(torch.version, 'hip', '7.1'), \
                     patch.object(torch.cuda, 'get_device_properties',
                                  return_value=SimpleNamespace(gcnArchName=arch)):
                    Krea2Driver().on_base_loaded(torch.nn.Linear(4, 4), 'nf4', 'cuda')
            with patch.object(torch.version, 'rocm', None, create=True), \
                 patch.object(torch.version, 'hip', None), \
                 patch.object(torch, '__version__', '2.12.0+cu130'), \
                 patch.object(torch.cuda, 'get_device_properties') as props:
                Krea2Driver().on_base_loaded(torch.nn.Linear(4, 4), 'nf4', 'cuda')
                props.assert_not_called()
            linear.assert_not_called()
            attention.assert_not_called()
            self.assertIsNone(sdpa._SDPA_CTX)

    def test_mlp_checkpoint_block_matches_original_outputs_and_gradients(self):
        from fizgig.krea2.model import SingleStreamBlock
        from rejected_checkpoint import install
        torch.manual_seed(42)
        original=SingleStreamBlock(128,1,1)
        changed=copy.deepcopy(original)
        holder=torch.nn.Module();holder.blocks=torch.nn.ModuleList([changed])
        install(holder)
        x=torch.randn(1,16,128,requires_grad=True)
        vec=torch.randn(1,128*6,requires_grad=True)*.1
        expected=original(x,vec,None)
        actual=changed(x,vec,None)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
        grad=torch.randn_like(actual)
        expected_grad=torch.autograd.grad(expected,[x,vec]+list(original.parameters()),grad)
        actual_grad=torch.autograd.grad(actual,[x,vec]+list(changed.parameters()),grad)
        for a,b in zip(actual_grad,expected_grad):
            torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-4)

    def test_attention_output_and_gradients_under_nested_checkpoint(self):
        torch.manual_seed(42)
        for shape in (None,(7,9),(2,1,7,9),(2,10,7,9)):
            inputs=[torch.randn(2,10,n,4,dtype=torch.float64,requires_grad=True) for n in (7,9,9)]
            mask=torch.rand(shape)>.2 if shape else None
            if mask is not None:
                mask[...,0]=True
                mask[...,-1,:]=False
            expected=F.scaled_dot_product_attention(*inputs,attn_mask=mask)
            g=torch.randn_like(expected)
            eg=torch.autograd.grad(expected,inputs,g)
            actual=checkpoint(lambda q,k,v:analytical_attention(q,k,v,mask),*inputs,use_reentrant=False)
            ag=torch.autograd.grad(actual,inputs,g)
            torch.testing.assert_close(actual,expected)
            for a,b in zip(ag,eg):torch.testing.assert_close(a,b)

    def test_attention_additive_mask_and_autocast(self):
        inputs=[torch.randn(1,4,n,8,requires_grad=True) for n in (5,7,7)]
        mask=torch.zeros(5,7);mask[:,-1]=float('-inf')
        expected=F.scaled_dot_product_attention(*inputs,attn_mask=mask)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            actual=analytical_attention(*inputs,attn_mask=mask)
        torch.testing.assert_close(actual,expected)

    def test_recomputed_norm_matches_original_gradients(self):
        torch.manual_seed(42)
        x=torch.randn(2,3,32,dtype=torch.bfloat16,requires_grad=True)
        scale=torch.randn(32,requires_grad=True)*.01
        expected=F.rms_norm(x.float(),(32,),weight=scale+1,eps=1e-5).bfloat16()
        g=torch.randn_like(expected)
        eg=torch.autograd.grad(expected,(x,scale),g)
        actual=_Norm.apply(x,scale,1e-5)
        ag=torch.autograd.grad(actual,(x,scale),g)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(ag[0].float(),eg[0].float(),rtol=.01,atol=.02)
        torch.testing.assert_close(ag[1],eg[1],rtol=1e-4,atol=1e-5)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
