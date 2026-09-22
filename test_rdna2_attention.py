import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fizgig.modules import rdna2_attention as rdna


class AttentionTests(unittest.TestCase):
    def test_masks_gradients_and_nested_checkpoint(self):
        torch.manual_seed(42)
        for mask_shape in (None, (7,9), (2,1,7,9), (2,10,7,9)):
            with self.subTest(mask_shape=mask_shape):
                tensors = [torch.randn(2,10,n,4,dtype=torch.float64,requires_grad=True) for n in (7,9,9)]
                mask = torch.rand(mask_shape) > .2 if mask_shape else None
                if mask is not None:mask[...,0] = True
                expected = F.scaled_dot_product_attention(*tensors,attn_mask=mask)
                grad = torch.randn_like(expected)
                expected_grads = torch.autograd.grad(expected,tensors,grad)
                actual = checkpoint(lambda q,k,v: rdna.head_chunk_attention(q,k,v,mask),*tensors,use_reentrant=False)
                actual_grads = torch.autograd.grad(actual,tensors,grad)
                torch.testing.assert_close(actual,expected)
                for a,b in zip(actual_grads,expected_grads):torch.testing.assert_close(a,b)

    def test_cpu_and_disabled_dispatch_do_not_probe_gpu(self):
        q = torch.randn(1,12,8,4)
        with patch.object(rdna,'_is_rdna2') as probe:
            torch.testing.assert_close(rdna.attention(q,q,q),F.scaled_dot_product_attention(q,q,q))
            with patch.dict(os.environ,{'FIZGIG_RDNA2_ATTENTION':'0'}):
                rdna.attention(q,q,q)
            probe.assert_not_called()

    def test_inference_has_no_checkpoint(self):
        q = torch.randn(1,10,7,4)
        with torch.no_grad(),patch.object(rdna,'checkpoint') as ckpt:
            torch.testing.assert_close(rdna.head_chunk_attention(q,q,q),F.scaled_dot_product_attention(q,q,q))
            ckpt.assert_not_called()


if __name__ == '__main__':
    unittest.main()
