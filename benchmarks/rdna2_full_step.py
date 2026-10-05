"""Synthetic full RAW-model LoRA steps; never reads or writes a user's training run."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
from rdna2_components import configure_runtime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', required=True)
    p.add_argument('--dit', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--variants', nargs='+',
                   choices=('baseline', 'heads2', 'analytical', 'tuned', 'tuned-sdpa', 'tiled', 'mlp-only'),
                   default=['baseline', 'heads2', 'analytical', 'tuned', 'tuned-sdpa', 'tiled'])
    p.add_argument('--repeats', type=int, default=2)
    args = p.parse_args()
    configure_runtime(args.runtime)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import torch
    import logging
    logging.basicConfig(level=logging.INFO)
    from fizgig.families.krea2 import KREA2
    from fizgig.families.lora import FamilyLoRA
    from fizgig.families.quant import load_base
    from fizgig.krea2.model import RMSNorm, SingleStreamBlock
    from fizgig.modules.rdna2_linear import is_rdna2_device
    if not is_rdna2_device('cuda') or torch.cuda.mem_get_info()[0] < 13*2**30:
        raise RuntimeError('Need an idle RDNA2 GPU with at least 13 GiB free')
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    driver = KREA2.load_driver()
    print('Loading the RAW model through streamed NF4...', flush=True)
    model, _ = load_base(driver,args.dit,'cuda','nf4',0)
    model.enable_gradient_checkpointing()
    original_forwards = {m:m.forward for m in model.modules() if isinstance(m,(RMSNorm,SingleStreamBlock))}
    net = FamilyLoRA(model,driver,device='cuda')
    net.add_trainable(16,16)
    parameters = net.parameters()
    initial = [p.detach().cpu().clone() for p in parameters]
    latent = torch.randn(1,16,88,88,device='cuda',dtype=torch.bfloat16)
    noise = torch.randn_like(latent)
    t = torch.tensor([.5],device='cuda')
    cond = {'hidden_states':torch.randn(1,64,12,2560,device='cuda',dtype=torch.bfloat16),
            'attention_mask':torch.ones(1,64,device='cuda',dtype=torch.bool)}
    result = {'torch':torch.__version__, 'device':torch.cuda.get_device_name(),
              'settings':{'resolution':[704,704],'text_tokens':64,'rank':16,'batch':1,
                          'optimizer':'AdamW8bit','previews':False,'ema':False}, 'measurements':[]}
    from bitsandbytes.optim import AdamW8bit
    flags = ('FIZGIG_RDNA2_HEADS','FIZGIG_RDNA2_ATTN_IMPL','FIZGIG_RDNA2_GEMM','FIZGIG_RDNA2_CHECKPOINT')
    for variant in args.variants:
        for flag in flags:
            os.environ.pop(flag,None)
        if variant != 'baseline':
            os.environ['FIZGIG_RDNA2_HEADS'] = '2'
        if variant in ('analytical', 'tuned', 'tiled', 'mlp-only'):
            os.environ['FIZGIG_RDNA2_ATTN_IMPL'] = 'analytical'
        if variant in ('tuned','mlp-only'):
            os.environ['FIZGIG_RDNA2_GEMM'] = 'tuned-fp16'
        if variant == 'tuned-sdpa':
            os.environ['FIZGIG_RDNA2_GEMM'] = 'tuned-fp16'
        if variant == 'tiled':
            os.environ['FIZGIG_RDNA2_GEMM'] = 'tiled-fp16'
        for module, forward in original_forwards.items():
            module.forward = forward
            if isinstance(module,SingleStreamBlock):
                module._handles_checkpointing = False
        driver.on_base_loaded(model,'nf4','cuda')
        if variant == 'mlp-only':
            from rejected_checkpoint import install
            install(model)
        with torch.no_grad():
            for param, value in zip(parameters,initial):
                param.copy_(value)
                param.grad = None
        model.train()
        optimizer = AdamW8bit(parameters,lr=1e-4)
        times = []
        losses = []
        gradients_finite = True
        print(f'Starting variant: {variant}',flush=True)
        try:
            for index in range(args.repeats+1):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if index == 1:
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                loss = driver.loss_at(model,latent,noise,t,cond)
                loss.backward()
                # Check finiteness outside the timed window.
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                gradients_finite &= all(bool(p.grad.isfinite().all()) for p in parameters if p.grad is not None)
                losses.append(loss.item())
                if index:
                    times.append(elapsed)
                print(json.dumps({'variant':variant,'index':index,'seconds':elapsed,'loss':loss.item()}),flush=True)
                del loss
            row = {'variant':variant,'median_s':statistics.median(times),'times_s':times,'losses':losses,
                   'gradients_finite':gradients_finite,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        except torch.OutOfMemoryError as error:
            row = {'variant':variant,'error':'out of memory'}
            print(str(error).splitlines()[0],flush=True)
        result['measurements'].append(row)
        Path(args.output).write_text(json.dumps(result,indent=2))
        print(json.dumps(row),flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        for param in parameters:
            param.grad = None
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
