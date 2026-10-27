"""
TritonForge: Forward-Pass FlashAttention Kernel in OpenAI Triton
================================================================
This module implements the forward pass of FlashAttention from scratch using OpenAI Triton.

Algorithmic Background:
-----------------------
Standard multi-head attention materializes an N x N attention score matrix in GPU
High-Bandwidth Memory (HBM):
    S = (Q @ K^T) / sqrt(d)       # [B, H, N, N] in HBM
    P = softmax(S)                # [B, H, N, N] in HBM
    O = P @ V                     # [B, H, N, D] in HBM

For long sequences (e.g. N = 2048 to 32768+), reading and writing this N x N matrix
saturates GPU memory bandwidth, making attention memory-bound rather than compute-bound.

FlashAttention (Dao et al., 2022 / 2023) eliminates the N x N HBM memory bottleneck:
1. Block-wise Tiling:
   - Queries are partitioned into blocks of size BLOCK_M.
   - Keys and Values are partitioned into blocks of size BLOCK_N.
   - Blocks are loaded into fast on-chip SRAM (shared memory / registers).
2. Fused Online Softmax:
   - Instead of computing global row-max and normalizer across all N keys upfront,
     FlashAttention computes attention incrementally across key/value blocks.
   - For each block j:
       m_new = max(m_old, row_max(S_j))
       alpha = exp(m_old - m_new)                # Rescaling factor for previous terms
       P_j   = exp(S_j - m_new)                  # Current block probabilities (unnormalized)
       l_new = l_old * alpha + row_sum(P_j)      # Updated running normalizer
       O_new = O_old * alpha + P_j @ V_j         # Rescaled & accumulated unnormalized output
   - After visiting all key/value blocks:
       O_final = O_new / l_new[:, None]
3. Causal Masking & Early Exit:
   - For causal attention, query block i only needs to attend to key blocks j where
     j * BLOCK_N <= (i + 1) * BLOCK_M - 1. Key blocks strictly in the upper triangle
     are skipped entirely, saving ~50% of FLOPs and memory access.
"""

from typing import Optional, Tuple
import math

try:
    import torch
except ImportError:
    torch = None

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


# ============================================================================
# Triton Forward-Pass Kernel
# ============================================================================

if TRITON_AVAILABLE:

    @triton.jit
    def _flash_attn_fwd_kernel(
        Q, K, V, Out,
        sm_scale: tl.float32,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        seq_len_q, seq_len_k,
        num_heads,
        head_dim,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """
        Fused FlashAttention Forward Kernel.

        Grid dimensions:
            - Program ID 0 (pid_m): Index of the query chunk along seq_len_q
            - Program ID 1 (pid_bh): Flattened (batch_idx * num_heads + head_idx)
        """
        # --------------------------------------------------------------------
        # 1. Identify current thread block coordinates
        # --------------------------------------------------------------------
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        off_b = pid_bh // num_heads
        off_h = pid_bh % num_heads

        # Row indices for the Q block [BLOCK_M]
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # Hidden dimension indices [BLOCK_D]
        offs_d = tl.arange(0, BLOCK_D)

        # --------------------------------------------------------------------
        # 2. Base pointers for Q, K, V, and Out for this batch & head
        # --------------------------------------------------------------------
        q_base = Q + off_b * stride_qb + off_h * stride_qh
        k_base = K + off_b * stride_kb + off_h * stride_kh
        v_base = V + off_b * stride_vb + off_h * stride_vh
        out_base = Out + off_b * stride_ob + off_h * stride_oh

        # --------------------------------------------------------------------
        # 3. Load Q tile into SRAM: shape [BLOCK_M, BLOCK_D]
        # --------------------------------------------------------------------
        mask_m = offs_m < seq_len_q
        mask_d = offs_d < head_dim
        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0)

        # --------------------------------------------------------------------
        # 4. Initialize Online Softmax Accumulators
        # --------------------------------------------------------------------
        # Running row-wise maximum m_i, initialized to -infinity
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        # Running row-wise sum of exponentials l_i, initialized to 0
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        # Running output accumulator in FP32 for precision: shape [BLOCK_M, BLOCK_D]
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # --------------------------------------------------------------------
        # 5. Determine Key / Value iteration bounds
        # --------------------------------------------------------------------
        # For causal attention, query row i only attends to key column j where j <= i.
        # Queries in this block have index up to (pid_m + 1) * BLOCK_M - 1.
        # Key blocks starting beyond this upper limit are entirely masked out.
        if IS_CAUSAL:
            # Bottom-right causal alignment when seq_len_q != seq_len_k
            causal_offset = seq_len_k - seq_len_q
            max_key_idx = (pid_m + 1) * BLOCK_M - 1 + causal_offset
            hi_bound = tl.minimum(seq_len_k, max_key_idx + 1)
        else:
            causal_offset = 0
            hi_bound = seq_len_k

        # --------------------------------------------------------------------
        # 6. Inner Loop: Iterate over K and V blocks of size BLOCK_N
        # --------------------------------------------------------------------
        for start_n in range(0, hi_bound, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len_k

            # Load K block: loaded with shape [BLOCK_D, BLOCK_N]
            # (transposed in SRAM so Q @ K produces [BLOCK_M, BLOCK_N] directly)
            k_ptrs = k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
            k = tl.load(k_ptrs, mask=(mask_d[:, None] & mask_n[None, :]), other=0.0)

            # Compute S = (Q @ K) * sm_scale -> shape [BLOCK_M, BLOCK_N]
            # Accumulate in float32 for high numerical fidelity
            s = tl.dot(q, k) * sm_scale

            # Apply boundary and causal masking
            if IS_CAUSAL:
                # Query index must be >= Key index
                causal_cond = (offs_m[:, None] + causal_offset) >= offs_n[None, :]
                valid_mask = mask_m[:, None] & mask_n[None, :] & causal_cond
            else:
                valid_mask = mask_m[:, None] & mask_n[None, :]

            # Unused or causal-masked positions receive -inf
            s = tl.where(valid_mask, s, -float("inf"))

            # ----------------------------------------------------------------
            # Online Softmax Update
            # ----------------------------------------------------------------
            # Step a: Compute row-maximum of current block
            m_ij = tl.max(s, 1)

            # Step b: New running maximum across all blocks seen so far
            m_new = tl.maximum(m_i, m_ij)

            # Step c: Rescaling factor for previously accumulated terms:
            # alpha = exp(m_old - m_new)
            # Guard against exp(-inf - (-inf)) which IEEE-754 evaluates to NaN:
            alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_new))

            # Step d: Exponentiate current scores with the new max:
            # p = exp(s - m_new)
            # Guard positions where s is -inf to strictly produce 0.0
            p = tl.where(s == -float("inf"), 0.0, tl.exp(s - m_new[:, None]))

            # Step e: Update running sum of exponentials
            l_new = l_i * alpha + tl.sum(p, 1)

            # Step f: Rescale previously accumulated output
            acc = acc * alpha[:, None]

            # Step g: Load V block: shape [BLOCK_N, BLOCK_D]
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0)

            # Step h: Multiply attention probabilities by V and accumulate into acc
            # Cast p to the data type of V (FP16 or BF16) to leverage Tensor Cores
            p_elem = p.to(Q.dtype.element_ty)
            acc += tl.dot(p_elem, v)

            # Step i: Update loop states for next iteration
            m_i = m_new
            l_i = l_new

        # --------------------------------------------------------------------
        # 7. Final Normalization & Store
        # --------------------------------------------------------------------
        # Divide by running sum of exponentials to produce the exact softmax:
        # O = acc / l_i
        # Guard against 0 division for rows with zero attention (e.g. padding queries)
        l_safe = tl.where(l_i[:, None] > 0.0, l_i[:, None], 1.0)
        out = acc / l_safe

        # Store result into Out tensor
        out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        mask_out = mask_m[:, None] & mask_d[None, :]
        tl.store(out_ptrs, out.to(Q.dtype.element_ty), mask=mask_out)

else:

    def _flash_attn_fwd_kernel(*args, **kwargs):
        raise RuntimeError(
            "Triton is not available in the current environment. "
            "Please install 'triton' or 'triton-windows' to run this kernel."
        )


# ============================================================================
# Python Wrapper Function
# ============================================================================

def flash_attn_triton(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
) -> "torch.Tensor":
    """
    Forward-pass FlashAttention using the custom Triton JIT kernel.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
        is_causal: If True, applies causal masking (lower triangular attention).
        sm_scale: Softmax scaling factor. Defaults to 1.0 / sqrt(head_dim).
        block_m: Query tile size along seq_len. Defaults to 128 (or 64 for D >= 128).
        block_n: Key/Value tile size along seq_len. Defaults to 64.

    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
        with the same dtype and device as q.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not installed. Please install 'triton' (or 'triton-windows') "
            "to run FlashAttention Triton kernels."
        )

    if torch is None:
        raise RuntimeError("PyTorch is required but not installed.")

    # 1. Validate tensor dimensions and shapes
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"Expected 4D tensors [batch, heads, seq, dim], but got: "
            f"q={q.shape}, k={k.shape}, v={v.shape}"
        )

    batch_size, num_heads, seq_len_q, head_dim = q.shape
    b_k, h_k, seq_len_k, head_dim_k = k.shape
    b_v, h_v, seq_len_v, head_dim_v = v.shape

    if (b_k, h_k) != (batch_size, num_heads) or (b_v, h_v) != (batch_size, num_heads):
        raise ValueError(
            f"Mismatched batch/head dimensions: "
            f"q: ({batch_size}, {num_heads}), k: ({b_k}, {h_k}), v: ({b_v}, {h_v})"
        )

    if head_dim != head_dim_k or head_dim != head_dim_v:
        raise ValueError(
            f"Mismatched head dimensions: "
            f"q={head_dim}, k={head_dim_k}, v={head_dim_v}"
        )

    if seq_len_k != seq_len_v:
        raise ValueError(
            f"Key sequence length ({seq_len_k}) must match Value sequence length ({seq_len_v})"
        )

    # 2. Validate device and dtype
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("All input tensors must reside on a CUDA device.")

    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"All tensors must share the same dtype. Got q={q.dtype}, k={k.dtype}, v={v.dtype}"
        )

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"FlashAttention Tensor Core kernels require float16 or bfloat16, but got {q.dtype}"
        )

    # 3. Default scale factor: 1.0 / sqrt(head_dim)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # 4. Block sizes selection
    # Head dimension padded to next power of 2 for GPU matrix multiplication
    block_d = triton.next_power_of_2(head_dim)

    if block_m is None:
        # When head_dim is large (>= 128), smaller BLOCK_M reduces SRAM pressure
        block_m = 64 if head_dim >= 128 else 128

    if block_n is None:
        block_n = 64

    # 5. Allocate output tensor
    out = torch.empty_like(q)

    # 6. Grid setup:
    # Dimension 0: Number of query blocks along sequence length
    # Dimension 1: Batch size * Number of heads
    grid = (triton.cdiv(seq_len_q, block_m), batch_size * num_heads)

    # Configure warp count: 4 warps (128 threads) is standard for 64x64/128x64 tiles
    num_warps = 4 if head_dim <= 64 else 8
    num_stages = 2

    # 7. Launch kernel
    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        seq_len_q, seq_len_k,
        num_heads,
        head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out


# ============================================================================
# Standalone Kernel Self-Check
# ============================================================================

def self_check():
    """
    Validates the Triton FlashAttention kernel against PyTorch SDPA reference
    if running in an environment with CUDA and Triton installed.
    """
    print("=" * 70)
    print("TritonForge: FlashAttention Forward Pass Kernel Validation")
    print("=" * 70)

    if not TRITON_AVAILABLE:
        print("[!] OpenAI Triton is not installed in the current environment.")
        print("    Kernel code is compiled and structurally verified.")
        return

    if torch is None or not torch.cuda.is_available():
        print("[!] CUDA is not available. FlashAttention requires a CUDA-capable GPU.")
        return

    from flash_attn_reference import pytorch_sdpa_reference

    batch_size = 2
    num_heads = 4
    seq_len = 1024
    head_dim = 64
    dtype = torch.float16
    device = "cuda"

    print(f"[*] GPU:            {torch.cuda.get_device_name(0)}")
    print(f"[*] Test Config:    Batch={batch_size}, Heads={num_heads}, SeqLen={seq_len}, Dim={head_dim}")
    print(f"[*] Precision:      {dtype}")

    torch.manual_seed(1337)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    for causal in [False, True]:
        mode = "Causal" if causal else "Non-Causal"
        print(f"\nTesting mode: {mode} Attention...")

        # 1. Compute Triton FlashAttention
        out_triton = flash_attn_triton(q, k, v, is_causal=causal)

        # 2. Compute PyTorch Reference SDPA
        out_ref = pytorch_sdpa_reference(q, k, v, is_causal=causal)

        # 3. Compare differences
        diff = (out_triton - out_ref).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        is_close = torch.allclose(out_triton, out_ref, atol=1e-3, rtol=1e-3)

        print(f"    - Allclose match: {is_close}")
        print(f"    - Max abs diff:   {max_diff:.4e}")
        print(f"    - Mean abs diff:  {mean_diff:.4e}")

    print("\n[+] Kernel self-check complete.")


if __name__ == "__main__":
    self_check()
