#!/usr/bin/env python3
"""
Benchmark script comparing SDPA vs FlashAttention-2 paths in nanoGPT.
Adapted from bench.py, runs both attention implementations back-to-back
and prints time/iter + MFU for both.
"""

import argparse
import time
import torch
import torch.nn as nn
from model import GPTConfig, GPT


def get_mfu(model, fwdbwd_per_iter, dt):
    """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
    N = sum(p.numel() for p in model.parameters() if not p.requires_grad or True)
    cfg = model.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
    flops_per_token = 6 * N + 12 * L * cfg.n_head * (cfg.n_embd // cfg.n_head) * cfg.block_size
    flops_per_fwdbwd = flops_per_token * cfg.block_size
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    flops_achieved = flops_per_iter * (1.0 / dt)  # per second
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    return flops_achieved / flops_promised


def run_benchmark(config, name, iterations=10, warmup=3):
    """Run benchmark for a given config."""
    torch.manual_seed(1337)
    
    # Create model
    gptconf = GPTConfig(**config)
    model = GPT(gptconf)
    model.train()
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Compile if on CUDA
    if torch.cuda.is_available():
        model = torch.compile(model)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)
    
    # Generate dummy data
    B = 12  # batch_size
    T = 1024  # block_size
    X = torch.randint(0, 50304, (B, 1024), device='cuda' if torch.cuda.is_available() else 'cpu')
    Y = torch.randint(0, 50304, (B, 1024), device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Warmup
    print(f"  Warming up {name}...")
    for _ in range(warmup):
        logits, loss = model(X, Y)
        loss.backward()
    
    # Benchmark
    print(f"  Benchmarking {name}...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    
    for i in range(iterations):
        logits, loss = model(X, Y)
        loss.backward()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    
    dt = (t1 - t0) / iterations
    mfu = model.estimate_mfu(1, dt)
    
    return dt, mfu


def main():
    parser = argparse.ArgumentParser(description='Benchmark FlashAttention vs SDPA in nanoGPT')
    parser.add_argument('--iterations', type=int, default=10, help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup iterations')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Benchmark will run on CPU (much slower).")
        print("FlashAttention kernels require CUDA and will not be used.")
        return
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print()
    
    # Base config
    base_config = dict(
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False,
        use_flash_attn=False,
        window_size=-1,
        alibi=False,
        flash_attn_deterministic=False,
    )
    
    # Test 1: SDPA (baseline)
    print("=" * 60)
    print("Test 1: PyTorch SDPA (use_flash_attn=False)")
    print("=" * 60)
    sdpa_config = base_config.copy()
    sdpa_config['use_flash_attn'] = False
    dt_sdpa, mfu_sdpa = run_benchmark(sdpa_config, "SDPA", iterations=10, warmup=3)
    print(f"  Time/iter: {dt_sdpa*1000:.2f} ms")
    print(f"  MFU: {mfu_sdpa*100:.2f}%")
    print()
    
    # Test 2: FlashAttention with defaults
    print("=" * 60)
    print("Test 2: FlashAttention-2 (use_flash_attn=True, defaults)")
    print("=" * 60)
    flash_config = base_config.copy()
    flash_config['use_flash_attn'] = True
    dt_flash, mfu_flash = run_benchmark(flash_config, "FlashAttention-2", iterations=10, warmup=3)
    print(f"  Time/iter: {dt_flash*1000:.2f} ms")
    print(f"  MFU: {mfu_flash*100:.2f}%")
    print()
    
    # Test 3: FlashAttention with sliding window
    print("=" * 60)
    print("Test 3: FlashAttention-2 + Sliding Window (window_size=256)")
    print("=" * 60)
    window_config = base_config.copy()
    window_config['use_flash_attn'] = True
    window_config['window_size'] = 256
    dt_window, mfu_window = run_benchmark(window_config, "FlashAttention-2 + Window", iterations=10, warmup=3)
    print(f"  Time/iter: {dt_window*1000:.2f} ms")
    print(f"  MFU: {mfu_window*100:.2f}%")
    print()
    
    # Test 4: FlashAttention with ALiBi
    print("=" * 60)
    print("Test 4: FlashAttention-2 + ALiBi (alibi=True)")
    print("=" * 60)
    alibi_config = base_config.copy()
    alibi_config['use_flash_attn'] = True
    alibi_config['alibi'] = True
    dt_alibi, mfu_alibi = run_benchmark(alibi_config, "FlashAttention-2 + ALiBi", iterations=10, warmup=3)
    print(f"  Time/iter: {dt_alibi*1000:.2f} ms")
    print(f"  MFU: {mfu_alibi*100:.2f}%")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<35} {'Time/iter (ms)':<15} {'MFU (%)':<10} {'Speedup':<10}")
    print("-" * 60)
    print(f"{'SDPA (baseline)':<35} {dt_sdpa*1000:<15.2f} {mfu_sdpa*100:<10.2f} {'1.00x':<10}")
    print(f"{'FlashAttention-2':<35} {dt_flash*1000:<15.2f} {mfu_flash*100:<10.2f} {dt_sdpa/dt_flash:.2f}x")
    print(f"{'FlashAttention-2 + Window (256)':<35} {dt_window*1000:<15.2f} {mfu_window*100:<10.2f} {dt_sdpa/dt_window:.2f}x")
    print(f"{'FlashAttention-2 + ALiBi':<35} {dt_alibi*1000:<15.2f} {mfu_alibi*100:<10.2f} {dt_sdpa/dt_alibi:.2f}x")
    print("=" * 60)
    
    if dt_flash < dt_sdpa:
        print(f"\nFlashAttention-2 is {dt_sdpa/dt_flash:.2f}x faster than SDPA!")
    else:
        print(f"\nSDPA is {dt_flash/dt_sdpa:.2f}x faster (or equal) - check hardware compatibility")


if __name__ == '__main__':
    main()