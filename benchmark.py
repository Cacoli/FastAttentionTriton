"""
TritonForge: FlashAttention Benchmark Suite
============================================
Benchmarks custom Triton FlashAttention against PyTorch's native SDPA
(`torch.nn.functional.scaled_dot_product_attention`) across scaling sequence lengths.

Key Features:
- Accurate GPU timing using `torch.cuda.Event` with warmup and repeated trials.
- Calculates latency (mean/median/std), Speedup, and effective compute throughput (TFLOPs/s).
- Formatted terminal summary table.
- High-resolution visual comparison plot saved as `benchmark_results.png`.
"""

from typing import List, Dict, Any, Tuple, Optional
import argparse
import sys
import math
import statistics

try:
    import torch
except ImportError:
    torch = None

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend suitable for headless/script execution
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    from flash_attn_reference import pytorch_sdpa_reference
    from flash_attn_triton import flash_attn_triton, TRITON_AVAILABLE
except ImportError:
    pytorch_sdpa_reference = None
    flash_attn_triton = None
    TRITON_AVAILABLE = False


def check_benchmark_environment() -> bool:
    """Verifies CUDA, PyTorch, and Triton availability."""
    if torch is None:
        print("[!] ERROR: PyTorch is not installed. Run 'pip install torch'.")
        return False

    if not torch.cuda.is_available():
        print("[!] ERROR: CUDA is not available. Benchmarking requires a CUDA-capable GPU.")
        return False

    if not TRITON_AVAILABLE:
        print("[!] ERROR: Triton is not available.")
        print("    On Linux / WSL2: pip install triton")
        print("    On Windows:      pip install triton-windows")
        return False

    return True


def compute_attention_flops(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool = False,
) -> float:
    """
    Computes theoretical floating point operations (FLOPs) for attention forward pass:
    - Q @ K^T: 2 * B * H * N * N * D operations
    - P @ V:   2 * B * H * N * N * D operations
    Total non-causal: 4 * B * H * N^2 * D
    Causal mask computes ~half the operations: 2 * B * H * N^2 * D
    """
    factor = 2 if is_causal else 4
    return factor * batch_size * num_heads * (seq_len ** 2) * head_dim


def benchmark_gpu_fn(
    fn,
    warmup: int = 25,
    repeats: int = 100,
) -> Tuple[float, float, float]:
    """
    Accurately benchmarks a GPU callable using torch.cuda.Event.

    Returns:
        Tuple of (median_ms, mean_ms, std_ms)
    """
    # 1. Warmup runs to allow JIT compilation, caching, and GPU clock stabilization
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 2. Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()

    latencies_ms = [
        start_events[i].elapsed_time(end_events[i]) for i in range(repeats)
    ]

    median_ms = statistics.median(latencies_ms)
    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    return median_ms, mean_ms, std_ms


def run_benchmark_suite(
    seq_lens: List[int],
    batch_size: int = 2,
    num_heads: int = 4,
    head_dim: int = 64,
    is_causal: bool = False,
    dtype: "torch.dtype" = None,
    warmup: int = 25,
    repeats: int = 100,
    output_plot: str = "benchmark_results.png",
) -> List[Dict[str, Any]]:
    """
    Runs benchmark comparing Triton FlashAttention vs PyTorch SDPA across sequence lengths.
    """
    if not check_benchmark_environment():
        return []

    if dtype is None:
        dtype = torch.float16

    device = "cuda"
    device_name = torch.cuda.get_device_name(0)

    print("=" * 95)
    print("TritonForge: FlashAttention vs PyTorch SDPA Benchmark")
    print("=" * 95)
    print(f"[*] GPU Device:      {device_name}")
    print(f"[*] Configuration:   Batch={batch_size}, Heads={num_heads}, Dim={head_dim}, "
          f"Causal={is_causal}, Precision={str(dtype).replace('torch.', '')}")
    print(f"[*] Timing:          {warmup} warmup iterations, {repeats} measurement repeats")
    print("-" * 95)

    header = (
        f"{'Seq Len':<9} | {'PyTorch (ms)':<14} | {'Triton (ms)':<14} | "
        f"{'Speedup':<9} | {'PyTorch TFLOP/s':<16} | {'Triton TFLOP/s':<16}"
    )
    print(header)
    print("-" * 95)

    records = []

    for n in seq_lens:
        # Create input tensors
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, n, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, n, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, n, head_dim, device=device, dtype=dtype)

        # Theoretical FLOPs
        flops = compute_attention_flops(batch_size, num_heads, n, head_dim, is_causal)

        # 1. Benchmark PyTorch SDPA
        def run_torch():
            return pytorch_sdpa_reference(q, k, v, is_causal=is_causal)

        torch_med, torch_mean, torch_std = benchmark_gpu_fn(run_torch, warmup=warmup, repeats=repeats)
        torch_tflops = (flops / (torch_med * 1e-3)) / 1e12

        # 2. Benchmark Triton FlashAttention
        def run_triton():
            return flash_attn_triton(q, k, v, is_causal=is_causal)

        triton_med, triton_mean, triton_std = benchmark_gpu_fn(run_triton, warmup=warmup, repeats=repeats)
        triton_tflops = (flops / (triton_med * 1e-3)) / 1e12

        # Speedup (PyTorch time / Triton time)
        speedup = torch_med / triton_med

        records.append({
            "seq_len": n,
            "torch_ms": torch_med,
            "triton_ms": triton_med,
            "speedup": speedup,
            "torch_tflops": torch_tflops,
            "triton_tflops": triton_tflops,
        })

        row = (
            f"{n:<9} | {torch_med:<14.3f} | {triton_med:<14.3f} | "
            f"{speedup:<8.2f}x | {torch_tflops:<16.2f} | {triton_tflops:<16.2f}"
        )
        print(row)

        # Free GPU memory between configurations
        del q, k, v
        torch.cuda.empty_cache()

    print("-" * 95)

    # Plot results if matplotlib is available
    if MATPLOTLIB_AVAILABLE and records:
        generate_benchmark_plot(records, device_name, is_causal, output_plot)
    elif not MATPLOTLIB_AVAILABLE:
        print("[!] Matplotlib is not installed. Plot was skipped (pip install matplotlib).")

    return records


def generate_benchmark_plot(
    records: List[Dict[str, Any]],
    gpu_name: str,
    is_causal: bool,
    output_filename: str = "benchmark_results.png",
):
    """
    Renders high-quality dual-panel performance comparison plots:
    - Left: Latency (ms) vs Sequence Length
    - Right: Speedup (x) and Throughput (TFLOPs/s) vs Sequence Length
    """
    seq_lens = [r["seq_len"] for r in records]
    torch_ms = [r["torch_ms"] for r in records]
    triton_ms = [r["triton_ms"] for r in records]
    speedups = [r["speedup"] for r in records]
    torch_tflops = [r["torch_tflops"] for r in records]
    triton_tflops = [r["triton_tflops"] for r in records]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # -------------------------------------------------------------------------
    # Panel 1: Execution Time (Latency)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(seq_lens, torch_ms, marker="o", linewidth=2.2, color="#E63946", label="PyTorch SDPA")
    ax1.plot(seq_lens, triton_ms, marker="s", linewidth=2.2, color="#1D3557", label="Triton FlashAttention")
    ax1.set_title("Forward-Pass Latency vs Sequence Length", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Sequence Length ($N$)", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Execution Time (ms)", fontsize=11, fontweight="semibold")
    ax1.set_xticks(seq_lens)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10, loc="upper left")

    # Annotate points with values
    for x, y in zip(seq_lens, triton_ms):
        ax1.annotate(f"{y:.2f}ms", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8.5, fontweight="semibold", color="#1D3557")

    # -------------------------------------------------------------------------
    # Panel 2: Compute Throughput & Speedup
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(seq_lens, triton_tflops, marker="^", linewidth=2.2, color="#2A9D8F", label="Triton TFLOP/s")
    ax2.plot(seq_lens, torch_tflops, marker="v", linewidth=2.0, linestyle="--", color="#F4A261", label="PyTorch TFLOP/s")
    ax2.set_title("Effective Throughput (TFLOPs/s)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Sequence Length ($N$)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Throughput (TFLOPs/s)", fontsize=11, fontweight="semibold")
    ax2.set_xticks(seq_lens)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Secondary y-axis for speedup
    ax2_twin = ax2.twinx()
    ax2_twin.plot(seq_lens, speedups, marker="D", linewidth=1.8, linestyle=":", color="#9B5DE5", label="Speedup (x)")
    ax2_twin.axhline(1.0, color="gray", linestyle="-.", linewidth=1.0, alpha=0.7)
    ax2_twin.set_ylabel("Speedup ($T_{PyTorch} / T_{Triton}$)", fontsize=11, fontweight="semibold", color="#9B5DE5")
    ax2_twin.tick_params(axis="y", labelcolor="#9B5DE5")

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9.5)

    # Title & Metadata
    causal_tag = "Causal" if is_causal else "Non-Causal"
    fig.suptitle(f"TritonForge: FlashAttention vs PyTorch SDPA [{gpu_name} | {causal_tag}]",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"[+] Performance plot saved successfully to: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="TritonForge FlashAttention Benchmarking Suite")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension (default: 64)")
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096],
        help="Sequence lengths to benchmark (default: 512 1024 2048 4096)"
    )
    parser.add_argument("--causal", action="store_true", help="Enable causal masking")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations (default: 25)")
    parser.add_argument("--repeats", type=int, default=100, help="Number of timed measurement repeats (default: 100)")
    parser.add_argument("--plot", type=str, default="benchmark_results.png", help="Output plot filename")

    args = parser.parse_args()

    run_benchmark_suite(
        seq_lens=args.seq_lens,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        is_causal=args.causal,
        warmup=args.warmup,
        repeats=args.repeats,
        output_plot=args.plot,
    )


if __name__ == "__main__":
    main()
