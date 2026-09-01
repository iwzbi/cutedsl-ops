"""FlashAttention decode — FP8, split-K + paged KV (Exercise 5).

Maps to hpc-ops ``D2 (smallm_fp8_qpertoken_perhead_kvpertensor_dim128_static)``.
The FP8 variant of exercise 3 — combines split-K decode + FP8 quant:

- ``QK=SS``: both Q and K operands for the QK^T matmul come from **S**MEM
  (SS-scope = SMEM x SMEM).  Q/K are loaded as fp8 and converted to bf16 in
  SMEM so the QK WGMMA is the bf16 atom.
- ``PV=SS``: P (A-operand) is staged to **S**MEM as fp8 after the x256 quant,
  V (B-operand) is in **S**MEM as fp8 (SS-scope fp8 WGMMA).
- Per-tensor K/V scales, per-token-per-head Q scale (passed through; the Q
  scale is part of the host ABI matching exercise 4).
- P pre-scale x256, fp8 cast, same as exercise 4.
- Split-K along KV dimension, same grid + LSE combine as exercise 3.

PREREQUISITE: Complete exercises 1-4.  Exercise 5 is the capstone — it
combines split-K (ex.3) with FP8 quant (ex.4).

Algorithm (per split CTA, two-pass like exercise 3 to avoid cross-TiledMma
broadcast issues)::

    Pass 1:  for j in n_start..n_end step BLK_N:
                 S = (Q_bf16 @ K_bf16^T) * softmax_scale * k_scale
                 online softmax -> m_local, l_local
    Pass 2:  for j in n_start..n_end step BLK_N:
                 S = (Q_bf16 @ K_bf16^T) * softmax_scale * k_scale
                 P = exp(S - m_local) / l_local
                 P_fp8 = (P * 256).to(fp8_e4m3)        # quantize
                 O += P_fp8 @ V_fp8                    # fp8 WGMMA
    LSE = m_local + log(l_local)
    store O_partial (unnormalized) + LSE  # host lse_combine merges splits
"""

from __future__ import annotations

import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
from cuda.bindings.driver import CUstream
from cutlass import cute
from cutlass.cute.nvgpu.warpgroup import OperandMajorMode, OperandSource
from cutlass.utils.layout import LayoutEnum


BLK_M = 64  # Q rows (decode M is tiny, padded)
BLK_N = 64  # KV cols per tile
D = 128  # head dim

NUM_STAGES = 4  # deeper pipeline target (fp8 loads are cheaper)

# Single warpgroup (no warp specialization) — same as exercise 3.
NUM_MMA_WAROGROUPS = 1
NUM_THREADS_PER_WAROGROUP = 128
NUM_THREADS = NUM_MMA_WAROGROUPS * NUM_THREADS_PER_WAROGROUP

P_SCALE = 256.0


@cute.kernel
def flash_attn_decode_fp8_kernel(
    mQ: cute.Tensor,  # (B*H, M, D)  fp8_e4m3
    mK_pages: cute.Tensor,  # (num_pages, H_kv, page_size, D)  fp8_e4m3
    mV_pages: cute.Tensor,  # (num_pages, H_kv, D, page_size)  fp8_e4m3
    mBlockTable: cute.Tensor,  # (B, max_blocks)  int32
    mO_partial: cute.Tensor,  # (kSplitK, B*H, M, D)  bf16
    mLSE_partial: cute.Tensor,  # (kSplitK, B*H, M)  fp32
    mQScale: cute.Tensor,  # (B*H, M, 1)  fp32
    tma_atom_v: cute.CopyAtom,
    tiled_mma_qk: cute.TiledMma,  # bf16 MMA (SS-scope: both from SMEM)
    tiled_mma_pv: cute.TiledMma,  # fp8 MMA (SS-scope: P and V from SMEM)
    r2s_tiled_copy_o: cute.TiledCopy,
    r2s_tiled_copy_p: cute.TiledCopy,
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sP_layout: cute.ComposedLayout,
    sO_layout: cute.ComposedLayout,
    kScale: cutlass.Constexpr[float],
    vScale: cutlass.Constexpr[float],
    page_size: cutlass.Constexpr[int],
    n_total: cutlass.Constexpr[int],
    split_size: cutlass.Constexpr[int],
    kSplitK: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    Dd: cutlass.Constexpr[int],
    tx_count_v: cutlass.Constexpr[int],
    shared_storage_cls: cutlass.Constexpr,
):
    """One CTA computes one split of (batch*head) along the KV dimension.

    Grid: (kSplitK, B*H, 1)
    """
    tid, _, _ = cute.arch.thread_idx()
    bid_split, bid_bh, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage_cls)

    sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
    sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
    sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
    sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
    sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
    mbar_ptr = storage.mbar.data_ptr()

    bid_b = bid_bh // H
    bid_h = bid_bh % H
    n_start = bid_split * split_size
    n_end = min(n_start + split_size, n_total)
    num_local_blocks = (n_end - n_start + BLK_N - 1) // BLK_N

    bf16 = cutlass.BFloat16
    f32 = cutlass.Float32
    fp8 = cutlass.Float8

    # Load Q once: fp8 gmem -> bf16 staging -> bf16 SMEM (stays for the split).
    gQ = cute.local_tile(mQ, (1, BLK_M, Dd), (bid_bh, 0, 0))
    gQ_2d = cute.group_modes(gQ, 0, 2)
    gQ_bf16 = cute.make_tensor_like(gQ_2d, bf16)
    cute.autovec_copy(gQ_2d, gQ_bf16)
    cute.autovec_copy(gQ_bf16, sQ)
    cute.arch.sync_threads()

    # QK partition (bf16 SS-scope: A=Q from sQ, B=K from sK, C=P in sP).
    thr_qk = tiled_mma_qk.get_slice(tid)
    tCsQ = thr_qk.partition_A(sQ)
    tCrQ = tiled_mma_qk.make_fragment_A(tCsQ)
    tCsS = thr_qk.partition_C(sP)
    tCrS = tiled_mma_qk.make_fragment_C(tCsS)

    # PV partition (fp8 SS-scope: A=P from sP, B=V from sV, C=O in sO).
    thr_pv = tiled_mma_pv.get_slice(tid)
    tCsP_a = thr_pv.partition_A(sP)
    tCrP_a = tiled_mma_pv.make_fragment_A(tCsP_a)
    tCsO = thr_pv.partition_C(sO)
    tCrO = tiled_mma_pv.make_fragment_C(tCsO)

    thr_r2s_p = r2s_tiled_copy_p.get_slice(tid)
    tDsP_r2s = thr_r2s_p.partition_D(sP)
    thr_r2s_o = r2s_tiled_copy_o.get_slice(tid)

    tCrO.fill(0.0)
    acc_c_shape = tCsS.shape
    reduced_shape = (acc_c_shape[0], acc_c_shape[1])
    m_i = cute.make_rmem_tensor(reduced_shape, f32)
    l_i = cute.make_rmem_tensor(reduced_shape, f32)
    m_i.fill(float("-inf"))
    l_i.fill(0.0)

    # Scale tensors (softmax_scale, per-tensor k_scale, P x256 pre-scale).
    scale_tensor = cute.make_rmem_tensor(acc_c_shape, f32)
    scale_tensor.fill(1.0 / (Dd**0.5))
    k_scale_tensor = cute.make_rmem_tensor(acc_c_shape, f32)
    k_scale_tensor.fill(kScale)
    p_scale_tensor = cute.make_rmem_tensor(acc_c_shape, f32)
    p_scale_tensor.fill(P_SCALE)

    num_k_blocks_qk = cute.size(tCrQ, mode=[2])
    num_k_blocks_pv = cute.size(tCrP_a, mode=[2])

    # Pass 1: compute m_local + l_local for this split (QK only, no V).
    for j_local in cutlass.range(num_local_blocks, unroll=1):
        j_global = n_start + j_local * BLK_N
        kv_idx = j_global // page_size
        page_id = mBlockTable[(bid_b, kv_idx)]

        # Load K as fp8, convert to bf16 in SMEM for the bf16 QK WGMMA.
        gK = cute.local_tile(mK_pages, (1, 1, page_size, Dd), (page_id, bid_h, 0, 0))
        gK_2d = cute.group_modes(gK, 0, 2)
        gK_bf16 = cute.make_tensor_like(gK_2d, bf16)
        cute.autovec_copy(gK_2d, gK_bf16)
        cute.autovec_copy(gK_bf16, sK)
        cute.arch.sync_threads()

        tCsK = thr_qk.partition_B(sK)
        tCrK = tiled_mma_qk.make_fragment_B(tCsK)

        cute.nvgpu.warpgroup.fence()
        tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        for k_idx in cutlass.range(num_k_blocks_qk, unroll_full=True):
            coord = (None, None, k_idx)
            cute.gemm(tiled_mma_qk, tCrS, tCrQ[coord], tCrK[coord], tCrS)
            tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)

        s_val = cute.math.mul(tCrS.load(), scale_tensor.load())
        s_val = cute.math.mul(s_val, k_scale_tensor.load())
        m_new = s_val.reduce(cute.ReductionOp.MAX, float("-inf"), (None, None, 1))
        m_new_b = m_new.broadcast_to(acc_c_shape)
        p_val = cute.math.exp(cute.math.sub(s_val, m_new_b))
        t_val = p_val.reduce(cute.ReductionOp.ADD, 0.0, (None, None, 1))
        alpha = cute.math.exp(cute.math.sub(m_i.load(), m_new))
        l_i.store(cute.math.add(cute.math.mul(l_i.load(), alpha), t_val))
        m_i.store(m_new)
        cute.arch.sync_threads()

    # Pass 2: O = sum(P_norm @ V) with P = exp(S - m_local) / l_local, fp8 quant.
    phase = 0
    for j_local in cutlass.range(num_local_blocks, unroll=1):
        j_global = n_start + j_local * BLK_N
        kv_idx = j_global // page_size
        page_id = mBlockTable[(bid_b, kv_idx)]

        # Reload K (fp8 -> bf16 SMEM) for the second QK pass.
        gK = cute.local_tile(mK_pages, (1, 1, page_size, Dd), (page_id, bid_h, 0, 0))
        gK_2d = cute.group_modes(gK, 0, 2)
        gK_bf16 = cute.make_tensor_like(gK_2d, bf16)
        cute.autovec_copy(gK_2d, gK_bf16)
        cute.autovec_copy(gK_bf16, sK)
        cute.arch.sync_threads()

        # Load V as fp8 via TMA into the 3D SMEM buffer (1, Dd, page_size).
        gV = cute.local_tile(mV_pages, (1, 1, Dd, page_size), (page_id, bid_h, 0, 0))
        gV_2d = cute.group_modes(gV, 0, 2)
        tAsV, tAgV = cute.nvgpu.cpasync.tma_partition(tma_atom_v, 0, cute.make_layout(1), sV, gV_2d)
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tx_count_v)
        cute.copy(tma_atom_v, tAgV[None, 0], tAsV[None, 0], tma_bar_ptr=mbar_ptr)
        cute.arch.mbarrier_wait(mbar_ptr, phase)
        phase = 1 - phase
        cute.arch.sync_threads()

        tCsK = thr_qk.partition_B(sK)
        tCrK = tiled_mma_qk.make_fragment_B(tCsK)
        tCsV = thr_pv.partition_B(sV)
        tCrV = tiled_mma_pv.make_fragment_B(tCsV)

        cute.nvgpu.warpgroup.fence()
        tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        for k_idx in cutlass.range(num_k_blocks_qk, unroll_full=True):
            coord = (None, None, k_idx)
            cute.gemm(tiled_mma_qk, tCrS, tCrQ[coord], tCrK[coord], tCrS)
            tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)

        # Softmax with global m, l (pre-normalized P), then x256 fp8 quant.
        s_val = cute.math.mul(tCrS.load(), scale_tensor.load())
        s_val = cute.math.mul(s_val, k_scale_tensor.load())
        m_global_b = m_i.load().broadcast_to(acc_c_shape)
        l_global_b = l_i.load().broadcast_to(acc_c_shape)
        p_val = cute.math.exp(cute.math.sub(s_val, m_global_b))
        p_val = cute.math.div(p_val, l_global_b)
        p_scaled = cute.math.mul(p_val, p_scale_tensor.load())
        tCrS.store(p_scaled)

        # r2s: stage P to sP as fp8 (A-operand for the fp8 PV WGMMA).
        tCrS_fp8 = cute.make_fragment_like(tCrS, fp8)
        tCrS_fp8.store(tCrS.load().to(fp8))
        tDrS_r2s = thr_r2s_p.retile(tCrS_fp8)
        cute.copy(r2s_tiled_copy_p, tDrS_r2s, tDsP_r2s)
        cute.arch.sync_threads()

        # PV matmul (fp8 SS-scope: A=P fp8 from sP, B=V fp8 from sV).
        cute.nvgpu.warpgroup.fence()
        for k_idx in cutlass.range(num_k_blocks_pv, unroll_full=True):
            coord = (None, None, k_idx)
            cute.gemm(tiled_mma_pv, tCrO, tCrP_a[coord], tCrV[coord], tCrO)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        cute.arch.sync_threads()

    # Epilogue: O *= vScale, store partial O (unnormalized) + LSE.
    v_scale_tensor = cute.make_rmem_tensor(tCrO.shape, f32)
    v_scale_tensor.fill(vScale)
    tCrO.store(cute.math.mul(tCrO.load(), v_scale_tensor.load()))

    tCrO_bf16 = cute.make_fragment_like(tCrO, bf16)
    tCrO_bf16.store(tCrO.load().to(bf16))
    tDrO = thr_r2s_o.retile(tCrO_bf16)
    gO = cute.local_tile(mO_partial, (1, 1, BLK_M, Dd), (bid_split, bid_bh, 0, 0))
    gO_2d = cute.group_modes(gO, 0, 2)
    tDgO = thr_r2s_o.partition_D(gO_2d)
    cute.copy(r2s_tiled_copy_o, tDrO, tDgO)

    # LSE = m + log(l), written per-thread (host lse_combine merges splits).
    lse_val = cute.math.add(m_i.load(), cute.math.log(l_i.load()))
    gLSE = cute.local_tile(mLSE_partial, (1, 1, reduced_shape[0]), (bid_split, bid_bh, 0))
    gLSE_2d = cute.group_modes(gLSE, 0, 2)
    cute.autovec_copy(lse_val, gLSE_2d)


@cute.jit
def flash_attn_decode_fp8(
    mQ: cute.Tensor,
    mK_pages: cute.Tensor,
    mV_pages: cute.Tensor,
    mBlockTable: cute.Tensor,
    mO_partial: cute.Tensor,
    mLSE_partial: cute.Tensor,
    mQScale: cute.Tensor,
    stream: CUstream,
    kScale: cutlass.Constexpr[float],
    vScale: cutlass.Constexpr[float],
    page_size: cutlass.Constexpr[int],
    n_total: cutlass.Constexpr[int],
    split_size: cutlass.Constexpr[int],
    kSplitK: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    Dd: cutlass.Constexpr[int],
):
    """Host entry: build bf16 (SS QK) + fp8 (SS PV) TiledMmas and launch."""
    bf16 = cutlass.BFloat16
    f32 = cutlass.Float32
    fp8 = cutlass.Float8
    mma_shape = (64, 16, 16)

    # QK^T: bf16 MMA, SS-scope (both operands from SMEM, K-major).
    op_qk = cute.nvgpu.warpgroup.MmaF16BF16Op(
        bf16, f32, mma_shape, OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K
    )
    tiled_mma_qk = cute.make_tiled_mma(cute.make_mma_atom(op_qk), (1, 4, 1))

    # PV: fp8 MMA, SS-scope (P and V both fp8 in SMEM, K-major).
    op_pv = cute.nvgpu.warpgroup.MmaF8Op(
        fp8, fp8, f32, mma_shape, OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K
    )
    tiled_mma_pv = cute.make_tiled_mma(cute.make_mma_atom(op_pv), (1, 8, 1))

    universal = cute.nvgpu.CopyUniversalOp()
    copy_atom_bf16 = cute.make_copy_atom(universal, bf16)
    copy_atom_fp8 = cute.make_copy_atom(universal, fp8)
    r2s_tiled_copy_o = cute.make_tiled_copy_C(copy_atom_bf16, tiled_mma_pv)
    r2s_tiled_copy_p = cute.make_tiled_copy_C(copy_atom_fp8, tiled_mma_qk)

    # SMEM layout atoms: Q/K bf16 (converted from fp8), V/P fp8, O bf16.
    q_atom = sm90_utils.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, Dd), bf16)
    k_atom = sm90_utils.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, Dd), bf16)
    v_atom = sm90_utils.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, fp8, page_size), fp8
    )
    p_atom = sm90_utils.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, fp8, BLK_N), fp8)

    sQ_layout = cute.tile_to_shape(q_atom, (BLK_M, Dd), order=(0, 1))
    sK_layout = cute.tile_to_shape(k_atom, (BLK_N, Dd), order=(0, 1))
    sV_layout = cute.tile_to_shape(v_atom, (Dd, page_size, 1), order=(0, 1, 2))
    sP_layout = cute.tile_to_shape(p_atom, (BLK_M, BLK_N), order=(0, 1))
    sO_layout = cute.tile_to_shape(q_atom, (BLK_M, Dd), order=(0, 1))

    sV_layout_one = cute.slice_(sV_layout, (None, None, 0))
    tma_atom_v, _ = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mV_pages, sV_layout_one, (1, 1, Dd, page_size)
    )
    tx_count_v = Dd * page_size * 1  # fp8 = 1 byte

    @cute.struct
    class SharedStorage:
        sQ: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sQ_layout)], 1024]
        sK: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sK_layout)], 1024]
        sV: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(sV_layout)], 1024]
        sP: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(sP_layout)], 1024]
        sO: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sO_layout)], 1024]
        mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 1], 8]

    BH = mQ.shape[0]
    H = BH  # Simplified: assume H_kv = H (harness handles GQA)
    num_threads = tiled_mma_qk.size
    grid = (kSplitK, BH, 1)

    flash_attn_decode_fp8_kernel(
        mQ,
        mK_pages,
        mV_pages,
        mBlockTable,
        mO_partial,
        mLSE_partial,
        mQScale,
        tma_atom_v,
        tiled_mma_qk,
        tiled_mma_pv,
        r2s_tiled_copy_o,
        r2s_tiled_copy_p,
        sQ_layout,
        sK_layout,
        sV_layout,
        sP_layout,
        sO_layout,
        kScale,
        vScale,
        page_size,
        n_total,
        split_size,
        kSplitK,
        H,
        M,
        Dd,
        tx_count_v,
        SharedStorage,
    ).launch(grid=grid, block=(num_threads, 1, 1), stream=stream)


__all__ = [
    "BLK_M",
    "BLK_N",
    "NUM_STAGES",
    "NUM_THREADS",
    "P_SCALE",
    "D",
    "flash_attn_decode_fp8",
    "flash_attn_decode_fp8_kernel",
]
