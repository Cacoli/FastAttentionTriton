"""
TritonForge: Correctness Verification Suite
===========================================
Validates the custom Triton forward FlashAttention kernel against PyTorch's
native scaled dot-product attention (`torch.nn.functional.scaled_dot_product_attention`)
across diverse tensor shapes, head dimensions, sequence lengths, and causal masking modes.

Test Matrix:
    - Batch Sizes:       [1, 2]
    - Number of Heads:   [4, 8]
    - Head Dimensions:   [64, 128]
    - Sequence Lengths:  [512, 1024, 2048]
    - Causal Modes:      [False, True]
"""

from typing import List, Dict, Any, Optional
import argparse
import sys
import time
import itertools

try:
    import torch
except ImportError:
    torch = None

try:
    from flash_attn_reference import pytorch_sdpa_reference
    from flash_attn_triton import flash_attn_triton, TRITON_AVAILABLE
except ImportError as e:
    pytorch_sdpa_reference = None
    flash_attn_triton = None
    TRITON_AVAILABLE = False


def check_environment() -> bool:
    """Verifies that PyTorch, Triton, and CUDA are ready for execution."""
    if torch is None:
        print("[!] ERROR: PyTorch is not installed. Install via 'pip install torch'.")
        return False

    if not torch.cuda.is_available():
        print("[!] ERROR: CUDA is not available. FlashAttention requires an NVIDIA GPU.")
        return False

    if not TRITON_AVAILABLE:
        print("[!] ERROR: Triton is not installed or available.")
        print("    On Linux / WSL2: pip install triton")
        print("    On Windows:      pip install triton-windows")
        return False

    return True


def run_single_test(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool,
    dtype: "torch.dtype" = None,
    device: str = "cuda",
    atol: float = 1e-3,
    rtol: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs a single correctness comparison between Triton FlashAttention and PyTorch SDPA.

    Returns:
        Dictionary containing configuration details and numerical metrics.
    """
    if dtype is None:
        dtype = torch.float16

    # Adjust tolerance for BF16 which has lower mantissa precision (7 bits vs 10 bits in FP16)
    if dtype == torch.bfloat16:
        atol = max(atol, 1e-2)
        rtol = max(rtol, 1e-2)

    torch.manual_seed(seed)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # 1. Reference PyTorch SDPA
    out_ref = pytorch_sdpa_reference(q, k, v, is_causal=is_causal)

    # 2. Custom Triton FlashAttention
    out_triton = flash_attn_triton(q, k, v, is_causal=is_causal)

    # 3. Numerical comparisons
    abs_diff = (out_triton - out_ref).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Check for NaN / Inf
    has_nan_triton = torch.isnan(out_triton).any().item()
    has_inf_triton = torch.isinf(out_triton).any().item()

    is_close = (
        not has_nan_triton
        and not has_inf_triton
        and torch.allclose(out_triton, out_ref, atol=atol, rtol=rtol)
    )

    return {
        "batch_size": batch_size,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "is_causal": is_causal,
        "dtype": str(dtype).replace("torch.", ""),
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "has_nan": has_nan_triton,
        "has_inf": has_inf_triton,
        "passed": is_close,
    }


def run_correctness_suite(
    batch_sizes: List[int],
    num_heads_list: List[int],
    head_dims: List[int],
    seq_lens: List[int],
    causal_modes: List[bool],
    dtype_str: str = "float16",
) -> bool:
    """
    Executes the full Cartesian product of test configurations and displays a summary table.
    """
    if not check_environment():
        return False

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    device = "cuda"

    print("=" * 88)
    print("TritonForge: FlashAttention Correctness Test Suite")
    print("=" * 88)
    print(f"[*] GPU Device:    {torch.cuda.get_device_name(0)}")
    print(f"[*] Precision:     {dtype}")
    print(f"[*] Configurations: {len(batch_sizes)} batch sizes x {len(num_heads_list)} heads "
          f"x {len(head_dims)} dims x {len(seq_lens)} seq lens x {len(causal_modes)} causal modes")
    total_cases = len(batch_sizes) * len(num_heads_list) * len(head_dims) * len(seq_lens) * len(causal_modes)
    print(f"[*] Total Tests:   {total_cases}")
    print("-" * 88)

    # Table Header
    header = (
        f"{'#':<4} | {'B':<3} | {'H':<3} | {'SeqLen':<7} | {'Dim':<5} | "
        f"{'Causal':<7} | {'Max Abs Diff':<13} | {'Mean Abs Diff':<13} | {'Status':<8}"
    )
    print(header)
    print("-" * 88)

    results: List[Dict[str, Any]] = []
    case_idx = 1
    start_time = time.time()

    configs = list(itertools.product(batch_sizes, num_heads_list, head_dims, seq_lens, causal_modes))

    for b, h, d, n, causal in configs:
        res = run_single_test(
            batch_size=b,
            num_heads=h,
            seq_len=n,
            head_dim=d,
            is_causal=causal,
            dtype=dtype,
            device=device,
        )
        results.append(res)

        status_str = "PASSED" if res["passed"] else "FAILED"
        causal_str = "True" if causal else "False"

        row = (
            f"{case_idx:<4} | {b:<3} | {h:<3} | {n:<7} | {d:<5} | "
            f"{causal_str:<7} | {res['max_diff']:<13.2e} | {res['mean_diff']:<13.2e} | {status_str:<8}"
        )
        print(row)
        case_idx += 1

    elapsed = time.time() - start_time
    total_passed = sum(1 for r in results if r["passed"])
    total_failed = total_cases - total_passed
    pass_rate = (total_passed / total_cases) * 100.0

    print("-" * 88)
    print(f"Summary: {total_passed}/{total_cases} Passed ({pass_rate:.1f}%) | "
          f"{total_failed} Failed | Total Time: {elapsed:.2f}s")
    print("=" * 88)

    if total_failed == 0:
        print("[+] All correctness tests passed successfully! FlashAttention kernel matches SDPA baseline.")
        return True
    else:
        print(f"[!] Warning: {total_failed} test case(s) failed numerical tolerance.")
        return False


def main():
    parser = argparse.ArgumentParser(description="TritonForge FlashAttention Correctness Test Suite")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a fast subset of configurations instead of full 48-case grid."
    )
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "bfloat16"], default="float16",
        help="Floating point precision to test (default: float16)."
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--num-heads", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--causal", type=int, nargs="+", default=[0, 1], help="0 for False, 1 for True")

    args = parser.parse_args()

    if args.quick:
        batch_sizes = [1]
        num_heads = [4]
        head_dims = [64, 128]
        seq_lens = [512, 1024]
        causal_modes = [False, True]
    else:
        batch_sizes = args.batch_sizes
        num_heads = args.num_heads
        head_dims = args.head_dims
        seq_lens = args.seq_lens
        causal_modes = [bool(c) for c in args.causal]

    success = run_correctness_suite(
        batch_sizes=batch_sizes,
        num_heads_list=num_heads,
        head_dims=head_dims,
        seq_lens=seq_lens,
        causal_modes=causal_modes,
        dtype_str=args.dtype,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
