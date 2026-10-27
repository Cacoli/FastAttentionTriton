"""
TritonForge: Reference Baseline for Scaled Dot-Product Attention (SDPA)
======================================================================
This module provides reference implementations of Scaled Dot-Product Attention:
1. `pytorch_sdpa_reference`: Uses PyTorch's native `torch.nn.functional.scaled_dot_product_attention`
   (which selects between FlashAttention, Memory-Efficient Attention, or Math kernels under the hood).
2. `naive_sdpa_reference`: Direct mathematical implementation of standard attention:
      Attention(Q, K, V) = softmax(Q @ K^T * scale + mask) @ V
   This serves as an explicit mathematical verification anchor and demonstrates
   the intermediate N x N attention matrix that FlashAttention avoids materializing in HBM.

Mathematical Background:
------------------------
Given queries Q, keys K, and values V with shapes:
    Q: [B, H, N_q, D]
    K: [B, H, N_k, D]
    V: [B, H, N_k, D_v]
where:
    B = Batch size
    H = Number of attention heads
    N_q, N_k = Query and Key/Value sequence lengths
    D = Head dimension (hidden dimension per head)

Standard Attention computes:
    1. Score matrix: S = (Q @ K^T) * scale            [B, H, N_q, N_k]
    2. Causal Mask (optional): S_ij = -inf for j > i
    3. Attention Weights: P = softmax(S, dim=-1)      [B, H, N_q, N_k]
    4. Output: O = P @ V                              [B, H, N_q, D_v]

FlashAttention Insight:
-----------------------
The NxN matrix S requires O(N^2) high-bandwidth memory (HBM) read/writes.
FlashAttention splits Q, K, V into blocks, loads blocks into fast GPU SRAM,
and computes attention incrementally via online softmax rescaling, reducing
memory traffic to O(N) without materializing P.
"""

from typing import Optional, Tuple
import math
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def get_default_scale(head_dim: int, scale: Optional[float] = None) -> float:
    """Return 1.0 / sqrt(head_dim) if scale is not specified."""
    return 1.0 / math.sqrt(head_dim) if scale is None else scale


def pytorch_sdpa_reference(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    dropout_p: float = 0.0,
) -> "torch.Tensor":
    """
    Computes Scaled Dot-Product Attention using PyTorch's optimized native function.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len_k, head_dim_v)
        is_causal: If True, applies causal lower-triangular masking.
        sm_scale: Softmax scaling factor. Defaults to 1.0 / sqrt(head_dim).
        dropout_p: Dropout probability applied to attention weights (default 0.0).

    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len_q, head_dim_v)
    """
    if torch is None:
        raise ImportError("PyTorch is required. Please install via 'pip install torch'.")

    head_dim = q.shape[-1]
    scale = get_default_scale(head_dim, sm_scale)

    # torch.nn.functional.scaled_dot_product_attention accepts scale directly
    out = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    return out


def naive_sdpa_reference(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
) -> "torch.Tensor":
    """
    Explicit mathematical computation of Scaled Dot-Product Attention:
        softmax((Q @ K.T) * scale + mask) @ V

    Materializes full (seq_len_q x seq_len_k) score matrix in memory.
    Used for verifying numerical precision against online softmax algorithms.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len_k, head_dim_v)
        is_causal: If True, masks upper triangle positions where key_idx > query_idx.
        sm_scale: Scaling factor for scores. Defaults to 1.0 / sqrt(head_dim).

    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len_q, head_dim_v)
    """
    if torch is None:
        raise ImportError("PyTorch is required. Please install via 'pip install torch'.")

    head_dim = q.shape[-1]
    scale = get_default_scale(head_dim, sm_scale)
    seq_len_q = q.shape[-2]
    seq_len_k = k.shape[-2]

    # Step 1: Batched matrix multiplication Q @ K^T -> [B, H, N_q, N_k]
    # Accumulate in float32 for numerical stability if inputs are half precision
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()

    scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * scale

    # Step 2: Causal Masking
    if is_causal:
        # mask is True where we want to keep, False where we want to mask (-inf)
        # For query row i and key col j, keep when j <= i
        causal_mask = torch.ones(
            (seq_len_q, seq_len_k), dtype=torch.bool, device=q.device
        ).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    # Step 3: Softmax along last dimension (keys dimension)
    probs = F.softmax(scores, dim=-1)

    # Handle NaN gracefully in case of empty sequence / all -inf rows
    probs = torch.nan_to_num(probs, nan=0.0)

    # Step 4: Multiply by V -> [B, H, N_q, D_v]
    out = torch.matmul(probs, v_f32)

    return out.to(dtype=q.dtype)


def verify_correctness(
    batch_size: int = 2,
    num_heads: int = 4,
    seq_len: int = 512,
    head_dim: int = 64,
    dtype: "torch.dtype" = None,
    device: Optional[str] = None,
    is_causal: bool = False,
) -> Tuple[bool, float, float]:
    """
    Compares `naive_sdpa_reference` against `pytorch_sdpa_reference`.

    Returns:
        Tuple of (is_close: bool, max_abs_diff: float, mean_abs_diff: float)
    """
    if torch is None:
        raise ImportError("PyTorch is not installed.")

    if dtype is None:
        dtype = torch.float32

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    torch.manual_seed(42)

    # Create synthetic query, key, value tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    out_pytorch = pytorch_sdpa_reference(q, k, v, is_causal=is_causal)
    out_naive = naive_sdpa_reference(q, k, v, is_causal=is_causal)

    abs_diff = (out_pytorch - out_naive).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Determine acceptable tolerance based on dtype
    rtol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    is_close = torch.allclose(out_pytorch, out_naive, rtol=rtol, atol=atol)

    return is_close, max_diff, mean_diff


def main():
    print("=" * 70)
    print("TritonForge: PyTorch SDPA Reference Baseline")
    print("=" * 70)

    if torch is None:
        print("[!] PyTorch is not yet installed in the current environment.")
        print("    Install dependencies via:")
        print("        pip install -r requirements.txt")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] PyTorch version: {torch.__version__}")
    print(f"[*] Target device:   {device}")
    if device == "cuda":
        print(f"[*] GPU Device:      {torch.cuda.get_device_name(0)}")
        print(f"[*] CUDA Capability: {torch.cuda.get_device_capability(0)}")

    configs = [
        {"desc": "Float32, Non-Causal", "dtype": torch.float32, "causal": False},
        {"desc": "Float32, Causal",     "dtype": torch.float32, "causal": True},
    ]

    # Test Half-precision if supported on device
    if device == "cuda" or (hasattr(torch, "float16") and device == "cpu"):
        configs.append({"desc": "Float16, Non-Causal", "dtype": torch.float16, "causal": False})
        configs.append({"desc": "Float16, Causal",     "dtype": torch.float16, "causal": True})

    if torch.cuda.is_bf16_supported() if device == "cuda" else False:
        configs.append({"desc": "BFloat16, Non-Causal", "dtype": torch.bfloat16, "causal": False})
        configs.append({"desc": "BFloat16, Causal",     "dtype": torch.bfloat16, "causal": True})

    print("\nRunning numerical consistency checks across configurations:")
    print("-" * 70)
    print(f"{'Configuration':<25} | {'Status':<8} | {'Max Abs Diff':<14} | {'Mean Abs Diff':<14}")
    print("-" * 70)

    for cfg in configs:
        is_close, max_diff, mean_diff = verify_correctness(
            batch_size=2,
            num_heads=4,
            seq_len=512,
            head_dim=64,
            dtype=cfg["dtype"],
            device=device,
            is_causal=cfg["causal"],
        )
        status_str = "PASSED" if is_close else "WARN"
        print(f"{cfg['desc']:<25} | {status_str:<8} | {max_diff:<14.2e} | {mean_diff:<14.2e}")

    print("-" * 70)
    print("[+] Reference baseline ready for benchmarking and Triton kernel validation.")


if __name__ == "__main__":
    main()
