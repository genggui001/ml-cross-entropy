from __future__ import annotations

from typing import Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from cut_cross_entropy.cce_backward import _mm_backward
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.tl_utils import b_bin_fn, is_triton_greater_or_equal_3_2_0
from cut_cross_entropy.utils import handle_reduction_none
from cut_cross_entropy.vocab_parallel import VocabParallelOptions
from cut_cross_entropy.vocab_parallel.utils import (
    vp_reduce_correct_logit,
    vp_reduce_e_grad_hook,
    vp_reduce_lse,
)


__all__ = ["LinearCrossEntropyKL", "linear_cross_entropy_kl"]


_LINEAR_CROSS_ENTROPY_KL_DOC = """Computes the fused student cross-entropy plus forward KL distillation loss
using logits from linear projections without materializing the full vocab logits.

Specifically, this computes

```python
student_logits = student_h @ student_c.T
teacher_logits = teacher_h @ teacher_c.T

student_ce_loss = F.cross_entropy(student_logits.float(), targets, reduction="none")
teacher_prob = F.softmax(teacher_logits.float(), dim=-1)
student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
kl_loss = torch.sum(teacher_prob * (teacher_log_prob - student_log_prob), dim=-1)

all_loss = student_ce_loss + alpha * kl_loss
```

without allocating the full `student_logits` / `teacher_logits` matrices in HBM.

Teacher tensors are treated as constants and never receive gradients.
This implementation follows the `cce_exact` spirit for precision: there is no gradient
filtering path, and both forward statistics and student-side gradient accumulation are
performed in fp32 before casting results back to the requested output dtype.
The current implementation requires `student_h` and `teacher_h` to have identical shapes,
and likewise requires `student_c` and `teacher_c` to have identical shapes.

:param student_h: Student hidden states. Shape (..., D)
:param student_c: Student classifier matrix. Shape (V, D)
:param targets: Student CE labels. Shape (...)
:param teacher_h: Teacher hidden states. Shape (..., D), must match `student_h`
:param teacher_c: Teacher classifier matrix. Shape (V, D), must match `student_c`
:param alpha: KL coefficient.
:param ignore_index: Labels equal to this value are skipped from both CE and KL.
:param reduction: Reduction over valid tokens. Supports `mean`, `sum`, and `none`.
:param return_components: If true, also returns `(student_ce_loss, kl_loss)` reduced the same way.
:param chunk_size: Vocab chunk size used by the custom backward pass. Larger values
    usually improve backward speed at the cost of higher temporary memory usage.
:param vocab_parallel_options: Optional aligned vocab-parallel shard definition for both
    `student_c` and `teacher_c`. When provided, both classifier matrices are expected to
    contain the same local vocab range `[start, stop)` for the same process group.
"""

_FWD_CONFIGS = [
    triton.Config({"BLOCK_B": 32, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 64, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 64, "BLOCK_V": 256, "BLOCK_D": 32}, num_warps=8, num_stages=3),
]

_BWD_CONFIGS = [
    triton.Config({"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_B": 128, "BLOCK_V": 256, "BLOCK_D": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 256, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 64, "BLOCK_V": 256, "BLOCK_D": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_B": 128, "BLOCK_V": 64, "BLOCK_D": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_B": 64, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=4),
]

@triton.autotune(
    configs=_FWD_CONFIGS,
    key=["num_rows", "local_vocab_size", "hidden_size", "vocab_per_split"],
)
@triton.jit
def _linear_cross_entropy_kl_forward_splits_kernel(
    student_h_ptr,
    student_c_ptr,
    labels_ptr,
    teacher_h_ptr,
    teacher_c_ptr,
    student_ce_max_ptr,
    student_ce_accu_ptr,
    teacher_kl_max_ptr,
    teacher_kl_accu_ptr,
    teacher_student_sum_ptr,
    teacher_teacher_sum_ptr,
    target_logit_ptr,
    num_rows,
    local_vocab_size,
    vocab_start,
    hidden_size,
    vocab_per_split,
    stride_student_h_row,
    stride_student_h_hidden,
    stride_student_c_vocab,
    stride_student_c_hidden,
    stride_teacher_h_row,
    stride_teacher_h_hidden,
    stride_teacher_c_vocab,
    stride_teacher_c_hidden,
    stride_student_ce_max_row,
    stride_student_ce_max_split,
    stride_student_ce_accu_row,
    stride_student_ce_accu_split,
    stride_teacher_kl_max_row,
    stride_teacher_kl_max_split,
    stride_teacher_kl_accu_row,
    stride_teacher_kl_accu_split,
    stride_teacher_student_sum_row,
    stride_teacher_student_sum_split,
    stride_teacher_teacher_sum_row,
    stride_teacher_teacher_sum_split,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(num_rows, BLOCK_B)
    pid_b = pid % num_pid_b
    pid_split = pid // num_pid_b

    split_start = pid_split * vocab_per_split
    split_stop = tl.minimum(split_start + vocab_per_split, local_vocab_size)

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    row_mask = offs_b < num_rows
    labels = tl.load(labels_ptr + offs_b, mask=row_mask, other=0).to(tl.int64)

    student_ce_max = tl.full((BLOCK_B,), -float("inf"), dtype=tl.float32)
    student_ce_accu = tl.zeros((BLOCK_B,), dtype=tl.float32)
    teacher_kl_max = tl.full((BLOCK_B,), -float("inf"), dtype=tl.float32)
    teacher_kl_accu = tl.zeros((BLOCK_B,), dtype=tl.float32)
    teacher_student_sum = tl.zeros((BLOCK_B,), dtype=tl.float32)
    teacher_teacher_sum = tl.zeros((BLOCK_B,), dtype=tl.float32)
    target_logit = tl.zeros((BLOCK_B,), dtype=tl.float32)

    for split_offset in range(0, vocab_per_split, BLOCK_V):
        local_offs_v = (split_start + split_offset + tl.arange(0, BLOCK_V)).to(tl.int64)
        global_offs_v = (vocab_start + local_offs_v).to(tl.int64)
        vocab_mask = local_offs_v < split_stop

        student_logits = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
        teacher_logits = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)

        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
        h_mask_base = row_mask[:, None]
        c_mask_base = vocab_mask[None, :]

        student_h_ptrs = (
            student_h_ptr
            + offs_b[:, None] * stride_student_h_row
            + offs_d[None, :] * stride_student_h_hidden
        )
        teacher_h_ptrs = (
            teacher_h_ptr
            + offs_b[:, None] * stride_teacher_h_row
            + offs_d[None, :] * stride_teacher_h_hidden
        )
        student_c_ptrs = (
            student_c_ptr
            + local_offs_v[None, :] * stride_student_c_vocab
            + offs_d[:, None] * stride_student_c_hidden
        )
        teacher_c_ptrs = (
            teacher_c_ptr
            + local_offs_v[None, :] * stride_teacher_c_vocab
            + offs_d[:, None] * stride_teacher_c_hidden
        )

        for _ in range(0, tl.cdiv(hidden_size, BLOCK_D)):
            h_mask = h_mask_base & (offs_d[None, :] < hidden_size)
            c_mask = c_mask_base & (offs_d[:, None] < hidden_size)

            student_h = tl.load(student_h_ptrs, mask=h_mask, other=0.0)
            teacher_h = tl.load(teacher_h_ptrs, mask=h_mask, other=0.0)
            student_c = tl.load(student_c_ptrs, mask=c_mask, other=0.0)
            teacher_c = tl.load(teacher_c_ptrs, mask=c_mask, other=0.0)

            student_logits = tl.dot(
                student_h,
                student_c,
                student_logits,
                input_precision="ieee",
            )
            teacher_logits = tl.dot(
                teacher_h,
                teacher_c,
                teacher_logits,
                input_precision="ieee",
            )

            offs_d += BLOCK_D
            student_h_ptrs += BLOCK_D * stride_student_h_hidden
            teacher_h_ptrs += BLOCK_D * stride_teacher_h_hidden
            student_c_ptrs += BLOCK_D * stride_student_c_hidden
            teacher_c_ptrs += BLOCK_D * stride_teacher_c_hidden

        student_logits = tl.where(vocab_mask[None, :], student_logits, -float("inf"))
        teacher_logits = tl.where(vocab_mask[None, :], teacher_logits, -float("inf"))

        old_student_ce_max = student_ce_max
        local_student_ce_max = tl.max(student_logits, axis=1)
        student_ce_max = tl.maximum(student_ce_max, local_student_ce_max)
        student_ce_accu = (
            student_ce_accu * tl.exp(old_student_ce_max - student_ce_max)
            + tl.sum(tl.exp(student_logits - student_ce_max[:, None]), axis=1)
        )

        student_logits_sum = tl.where(vocab_mask[None, :], student_logits, 0.0)
        teacher_logits_sum = tl.where(vocab_mask[None, :], teacher_logits, 0.0)

        old_teacher_kl_max = teacher_kl_max
        local_teacher_kl_max = tl.max(teacher_logits, axis=1)
        teacher_kl_max = tl.maximum(teacher_kl_max, local_teacher_kl_max)
        teacher_coeff = tl.exp(old_teacher_kl_max - teacher_kl_max)
        teacher_exp = tl.exp(teacher_logits - teacher_kl_max[:, None])
        teacher_kl_accu = teacher_kl_accu * teacher_coeff + tl.sum(teacher_exp, axis=1)
        teacher_student_sum = teacher_student_sum * teacher_coeff + tl.sum(
            teacher_exp * student_logits_sum, axis=1
        )
        teacher_teacher_sum = teacher_teacher_sum * teacher_coeff + tl.sum(
            teacher_exp * teacher_logits_sum, axis=1
        )

        label_mask = (
            row_mask[:, None]
            & vocab_mask[None, :]
            & (labels[:, None] == global_offs_v[None, :])
        )
        target_logit += tl.sum(tl.where(label_mask, student_logits, 0.0), axis=1)

    split_ptr = pid_split.to(tl.int64)
    student_ce_max_dst = (
        student_ce_max_ptr + offs_b * stride_student_ce_max_row + split_ptr * stride_student_ce_max_split
    )
    student_ce_accu_dst = (
        student_ce_accu_ptr
        + offs_b * stride_student_ce_accu_row
        + split_ptr * stride_student_ce_accu_split
    )
    teacher_kl_max_dst = (
        teacher_kl_max_ptr + offs_b * stride_teacher_kl_max_row + split_ptr * stride_teacher_kl_max_split
    )
    teacher_kl_accu_dst = (
        teacher_kl_accu_ptr
        + offs_b * stride_teacher_kl_accu_row
        + split_ptr * stride_teacher_kl_accu_split
    )
    teacher_student_sum_dst = (
        teacher_student_sum_ptr
        + offs_b * stride_teacher_student_sum_row
        + split_ptr * stride_teacher_student_sum_split
    )
    teacher_teacher_sum_dst = (
        teacher_teacher_sum_ptr
        + offs_b * stride_teacher_teacher_sum_row
        + split_ptr * stride_teacher_teacher_sum_split
    )

    tl.store(student_ce_max_dst, student_ce_max, mask=row_mask)
    tl.store(student_ce_accu_dst, student_ce_accu, mask=row_mask)
    tl.store(teacher_kl_max_dst, teacher_kl_max, mask=row_mask)
    tl.store(teacher_kl_accu_dst, teacher_kl_accu, mask=row_mask)
    tl.store(teacher_student_sum_dst, teacher_student_sum, mask=row_mask)
    tl.store(teacher_teacher_sum_dst, teacher_teacher_sum, mask=row_mask)

    has_target_in_split = (
        row_mask
        & (labels >= (vocab_start + split_start))
        & (labels < (vocab_start + split_stop))
    )
    tl.store(target_logit_ptr + offs_b, target_logit, mask=has_target_in_split)


def _reduce_lse_stats(partial_max: torch.Tensor, partial_accu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row_max = partial_max.max(dim=1).values
    row_scale = torch.exp(partial_max - row_max.unsqueeze(1))
    row_accu = torch.sum(row_scale * partial_accu, dim=1)
    return row_max, row_accu.log()


def _reduce_teacher_stats(
    partial_max: torch.Tensor,
    partial_accu: torch.Tensor,
    partial_student_sum: torch.Tensor,
    partial_teacher_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row_max = partial_max.max(dim=1).values
    row_scale = torch.exp(partial_max - row_max.unsqueeze(1))
    row_accu = torch.sum(row_scale * partial_accu, dim=1)
    row_student_sum = torch.sum(row_scale * partial_student_sum, dim=1)
    row_teacher_sum = torch.sum(row_scale * partial_teacher_sum, dim=1)
    return row_max, row_accu.log(), row_student_sum / row_accu, row_teacher_sum / row_accu


def _vp_reduce_teacher_expectation(
    local_expectation: torch.Tensor,
    local_lse: torch.Tensor,
    global_lse: torch.Tensor,
    pg: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    weighted_expectation = local_expectation * torch.exp(local_lse - global_lse)
    torch.distributed.all_reduce(weighted_expectation, group=pg)
    return weighted_expectation


def _kl_back_block_d(args) -> int:
    return 2 * args["BLOCK_D"]


def _linear_cross_entropy_kl_backward_kernel(
    E,
    C,
    TeacherH,
    TeacherC,
    StudentCeLSE,
    TeacherKlLSE,
    CeCoeff,
    KlCoeff,
    Targets,
    dE,
    dEC,
    dELocks,
    dC,
    dCC,
    dCLocks,
    B,
    D,
    V,
    n_de_locks_0,
    n_de_locks_1,
    n_dc_locks_0,
    n_dc_locks_1,
    stride_student_h_row,
    stride_student_h_hidden,
    stride_student_c_vocab,
    stride_student_c_hidden,
    stride_teacher_h_row,
    stride_teacher_h_hidden,
    stride_teacher_c_vocab,
    stride_teacher_c_hidden,
    vocab_start,
    USE_KAHAN_E,
    USE_KAHAN_C,
    USE_DE,
    USE_DC,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MM_BACK_BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    MM_BACK_EVEN_D: tl.constexpr,
    KAHAN_E: tl.constexpr,
    KAHAN_C: tl.constexpr,
    COMPUTE_DE: tl.constexpr,
    COMPUTE_DC: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_v_chunks = tl.cdiv(V, BLOCK_V)
    num_v_in_group = GROUP_B * num_v_chunks
    group_id = pid // num_v_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = (first_pid_b + ((pid % num_v_in_group) % group_size_b)).to(tl.int64)
    pid_v = ((pid % num_v_in_group) // group_size_b).to(tl.int64)

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)

    student_h_ptrs = (
        E
        + offs_b[:, None] * stride_student_h_row
        + offs_d[None, :] * stride_student_h_hidden
    )
    teacher_h_ptrs = (
        TeacherH
        + offs_b[:, None] * stride_teacher_h_row
        + offs_d[None, :] * stride_teacher_h_hidden
    )
    student_c_ptrs = (
        C
        + offs_v[None, :] * stride_student_c_vocab
        + offs_d[:, None] * stride_student_c_hidden
    )
    teacher_c_ptrs = (
        TeacherC
        + offs_v[None, :] * stride_teacher_c_vocab
        + offs_d[:, None] * stride_teacher_c_hidden
    )

    student_logits = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    teacher_logits = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)

    for d in range(0, tl.cdiv(D, BLOCK_D)):
        h_mask = offs_b[:, None] < B
        if not EVEN_D:
            h_mask = h_mask & (offs_d[None, :] < (D - d * BLOCK_D))

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))

        student_h = tl.load(student_h_ptrs, mask=h_mask, other=0.0)
        teacher_h = tl.load(teacher_h_ptrs, mask=h_mask, other=0.0)
        student_c = tl.load(student_c_ptrs, mask=c_mask, other=0.0)
        teacher_c = tl.load(teacher_c_ptrs, mask=c_mask, other=0.0)

        student_logits = tl.dot(student_h, student_c, student_logits, input_precision=DOT_PRECISION)
        teacher_logits = tl.dot(teacher_h, teacher_c, teacher_logits, input_precision=DOT_PRECISION)

        student_h_ptrs += BLOCK_D * stride_student_h_hidden
        teacher_h_ptrs += BLOCK_D * stride_teacher_h_hidden
        student_c_ptrs += BLOCK_D * stride_student_c_hidden
        teacher_c_ptrs += BLOCK_D * stride_teacher_c_hidden

    student_ce_lse = tl.load(StudentCeLSE + offs_b, mask=offs_b < B, other=float("inf"))
    teacher_kl_lse = tl.load(TeacherKlLSE + offs_b, mask=offs_b < B, other=float("inf"))
    ce_coeff = tl.load(CeCoeff + offs_b, mask=offs_b < B, other=0.0)
    kl_coeff = tl.load(KlCoeff + offs_b, mask=offs_b < B, other=0.0)
    targets = tl.load(Targets + offs_b, mask=offs_b < B, other=-1)

    student_logits = tl.where(offs_v[None, :] < V, student_logits, -float("inf"))
    teacher_logits = tl.where(offs_v[None, :] < V, teacher_logits, -float("inf"))

    student_prob = tl.exp(student_logits - student_ce_lse[:, None])
    d_accum = student_prob * ce_coeff[:, None]
    global_offs_v = (vocab_start + offs_v).to(tl.int64)
    is_target = (
        (offs_b[:, None] < B)
        & (offs_v[None, :] < V)
        & (targets[:, None] == global_offs_v[None, :])
    )
    d_accum += tl.where(is_target, -ce_coeff[:, None], 0.0)

    teacher_kl_prob = tl.exp(teacher_logits - teacher_kl_lse[:, None])
    d_accum += (student_prob - teacher_kl_prob) * kl_coeff[:, None]

    d_accum_fp32 = d_accum
    d_accum_lowp = d_accum_fp32.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")

    if COMPUTE_DE:
        lock_offset = (pid_b // tl.cdiv(B, BLOCK_B * n_de_locks_0)) * n_de_locks_1
        _mm_backward(
            d_accum_lowp,
            dE + (offs_b[:, None] * stride_student_h_row),
            dEC + (offs_b[:, None] * stride_student_h_row) if KAHAN_E else None,
            offs_b[:, None] < B,
            dELocks + lock_offset,
            n_de_locks_1,
            C + offs_v[:, None] * stride_student_c_vocab,
            offs_v[:, None] < V,
            stride_student_h_hidden,
            stride_student_c_hidden,
            D,
            MM_BACK_BLOCK_D,
            MM_BACK_EVEN_D,
            KAHAN_E,
            DOT_PRECISION,
        )

    if COMPUTE_DC:
        lock_offset = (pid_v // tl.cdiv(V, BLOCK_V * n_dc_locks_0)) * n_dc_locks_1
        _mm_backward(
            tl.trans(d_accum_lowp),
            dC + (offs_v[:, None] * stride_student_c_vocab),
            dCC + (offs_v[:, None] * stride_student_c_vocab) if KAHAN_C else None,
            offs_v[:, None] < V,
            dCLocks + lock_offset,
            n_dc_locks_1,
            E + (offs_b[:, None] * stride_student_h_row),
            offs_b[:, None] < B,
            stride_student_c_hidden,
            stride_student_h_hidden,
            D,
            MM_BACK_BLOCK_D,
            MM_BACK_EVEN_D,
            KAHAN_C,
            DOT_PRECISION,
        )


_linear_cross_entropy_kl_backward_kernel = triton.jit(_linear_cross_entropy_kl_backward_kernel)
_linear_cross_entropy_kl_backward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
        "MM_BACK_BLOCK_D": lambda args: _kl_back_block_d(args),
        "MM_BACK_EVEN_D": lambda args: (args["D"] % _kl_back_block_d(args)) == 0,
        "GROUP_B": lambda args: 8,
        "COMPUTE_DE": lambda args: args["USE_DE"],
        "COMPUTE_DC": lambda args: args["USE_DC"],
        "KAHAN_E": lambda args: args["USE_KAHAN_E"],
        "KAHAN_C": lambda args: args["USE_KAHAN_C"],
        "DOT_PRECISION": lambda args: "tf32"
        if torch.get_float32_matmul_precision() == "high"
        else "ieee",
    }
)(_linear_cross_entropy_kl_backward_kernel)
_linear_cross_entropy_kl_backward_kernel = triton.autotune(  # type: ignore
    configs=_BWD_CONFIGS,
    key=["V", "D", "B_BIN"],
    reset_to_zero=["dE", "dC", "dEC", "dCC"],
)(_linear_cross_entropy_kl_backward_kernel)


def _linear_cross_entropy_kl_backward_launcher(
    ce_coeff: torch.Tensor,
    kl_coeff: torch.Tensor,
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    student_ce_lse: torch.Tensor,
    teacher_kl_lse: torch.Tensor,
    *,
    vocab_start: int,
    need_student_h: bool,
    need_student_c: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not is_triton_greater_or_equal_3_2_0():
        assert student_h.dtype in (torch.float16, torch.bfloat16)
        assert student_c.dtype in (torch.float16, torch.bfloat16)
        can_use_fp32_accum = False
    else:
        can_use_fp32_accum = True

    ce_coeff = ce_coeff.contiguous()
    kl_coeff = kl_coeff.contiguous()
    targets = targets.contiguous()
    student_ce_lse = student_ce_lse.contiguous()
    teacher_kl_lse = teacher_kl_lse.contiguous()

    de_dtype = torch.float32 if (need_student_h and can_use_fp32_accum) else None
    dc_dtype = torch.float32 if (need_student_c and can_use_fp32_accum) else None

    use_kahan_e = need_student_h and not can_use_fp32_accum
    use_kahan_c = need_student_c and not can_use_fp32_accum

    dE = torch.zeros_like(student_h, dtype=de_dtype) if need_student_h else student_h.new_zeros((1, 1))
    dC = torch.zeros_like(student_c, dtype=dc_dtype) if need_student_c else student_c.new_zeros((1, 1))

    dEC = torch.zeros_like(student_h) if use_kahan_e else student_h.new_zeros((1, 1))
    dCC = torch.zeros_like(student_c) if use_kahan_c else student_c.new_zeros((1, 1))

    n_d_locks = triton.cdiv(student_c.size(1), 64)
    dELocks = (
        student_h.new_zeros((triton.cdiv(student_h.size(0), 128), n_d_locks), dtype=torch.int32)
        if need_student_h
        else student_h.new_zeros((1, 1), dtype=torch.int32)
    )
    dCLocks = (
        student_c.new_zeros((triton.cdiv(student_c.size(0), 128), n_d_locks), dtype=torch.int32)
        if need_student_c
        else student_c.new_zeros((1, 1), dtype=torch.int32)
    )

    def grid(meta):
        return (triton.cdiv(student_h.size(0), meta["BLOCK_B"]) * triton.cdiv(student_c.size(0), meta["BLOCK_V"]),)

    _linear_cross_entropy_kl_backward_kernel[grid](
        student_h,
        student_c,
        teacher_h,
        teacher_c,
        student_ce_lse,
        teacher_kl_lse,
        ce_coeff,
        kl_coeff,
        targets,
        dE,
        dEC,
        dELocks,
        dC,
        dCC,
        dCLocks,
        student_h.size(0),
        student_h.size(1),
        student_c.size(0),
        None if dELocks is None else dELocks.size(0),
        None if dELocks is None else dELocks.size(1),
        None if dCLocks is None else dCLocks.size(0),
        None if dCLocks is None else dCLocks.size(1),
        student_h.stride(0),
        student_h.stride(1),
        student_c.stride(0),
        student_c.stride(1),
        teacher_h.stride(0),
        teacher_h.stride(1),
        teacher_c.stride(0),
        teacher_c.stride(1),
        vocab_start=vocab_start,
        USE_KAHAN_E=use_kahan_e,
        USE_KAHAN_C=use_kahan_c,
        USE_DE=need_student_h,
        USE_DC=need_student_c,
        B_BIN=b_bin_fn(student_h.size(0)),
    )

    if need_student_h:
        dE = dE.to(dtype=student_h.dtype)
    else:
        dE = None

    if need_student_c:
        dC = dC.to(dtype=student_c.dtype)
    else:
        dC = None
    return dE, dC


def _validate_inputs(
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    vocab_parallel_options: VocabParallelOptions | None,
) -> None:
    if student_h.ndim < 2 or teacher_h.ndim < 2:
        raise ValueError("student_h and teacher_h must have shape (..., hidden)")
    if student_h.shape != teacher_h.shape:
        raise ValueError(
            f"student_h and teacher_h must have identical shape: {student_h.shape} vs {teacher_h.shape}"
        )
    if student_h.shape[:-1] != targets.shape:
        raise ValueError(
            f"student_h leading shape must match targets: {student_h.shape[:-1]} vs {targets.shape}"
        )
    if student_c.ndim != 2 or teacher_c.ndim != 2:
        raise ValueError("student_c and teacher_c must have shape (vocab, hidden)")
    if student_c.shape != teacher_c.shape:
        raise ValueError(
            f"student_c and teacher_c must have identical shape: {student_c.shape} vs {teacher_c.shape}"
        )
    if student_h.shape[-1] != student_c.shape[1]:
        raise ValueError(
            f"student hidden size mismatch: {student_h.shape[-1]} vs {student_c.shape[1]}"
        )
    if vocab_parallel_options is None:
        if student_c.shape[0] <= 0:
            raise ValueError("vocab size must be positive")
    else:
        expected_v_dim_size = vocab_parallel_options.stop - vocab_parallel_options.start
        if student_c.size(0) != expected_v_dim_size:
            raise ValueError(
                f"Expected student_c.size(0) to be {expected_v_dim_size}, got {student_c.size(0)}."
            )
        if teacher_c.size(0) != expected_v_dim_size:
            raise ValueError(
                f"Expected teacher_c.size(0) to be {expected_v_dim_size}, got {teacher_c.size(0)}."
            )
    tensors = (student_h, student_c, teacher_h, teacher_c)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("linear_cross_entropy_kl expects CUDA tensors")
    devices = {t.device for t in tensors}
    if len(devices) != 1:
        raise ValueError("student/teacher tensors must be on the same CUDA device")


class _LinearCrossEntropyKLFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        student_h: torch.Tensor,
        student_c: torch.Tensor,
        targets: torch.Tensor,
        teacher_h: torch.Tensor,
        teacher_c: torch.Tensor,
        alpha: float,
        chunk_size: int,
        vocab_parallel_options: VocabParallelOptions | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_rows = targets.numel()
        local_vocab_size = student_c.size(0)
        vocab_start = 0 if vocab_parallel_options is None else vocab_parallel_options.start
        vocab_per_split = min(1024, max(128, local_vocab_size))
        num_splits = max(1, triton.cdiv(local_vocab_size, vocab_per_split))

        student_ce_max = student_h.new_full((num_rows, num_splits), -torch.inf, dtype=torch.float32)
        student_ce_accu = student_h.new_zeros((num_rows, num_splits), dtype=torch.float32)
        teacher_kl_max = teacher_h.new_full((num_rows, num_splits), -torch.inf, dtype=torch.float32)
        teacher_kl_accu = teacher_h.new_zeros((num_rows, num_splits), dtype=torch.float32)
        teacher_student_sum = teacher_h.new_zeros((num_rows, num_splits), dtype=torch.float32)
        teacher_teacher_sum = teacher_h.new_zeros((num_rows, num_splits), dtype=torch.float32)
        target_logit = student_h.new_zeros((num_rows,), dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(num_rows, meta["BLOCK_B"]) * num_splits,)
        _linear_cross_entropy_kl_forward_splits_kernel[grid](
            student_h,
            student_c,
            targets,
            teacher_h,
            teacher_c,
            student_ce_max,
            student_ce_accu,
            teacher_kl_max,
            teacher_kl_accu,
            teacher_student_sum,
            teacher_teacher_sum,
            target_logit,
            num_rows,
            local_vocab_size,
            vocab_start,
            student_h.size(1),
            vocab_per_split,
            student_h.stride(0),
            student_h.stride(1),
            student_c.stride(0),
            student_c.stride(1),
            teacher_h.stride(0),
            teacher_h.stride(1),
            teacher_c.stride(0),
            teacher_c.stride(1),
            student_ce_max.stride(0),
            student_ce_max.stride(1),
            student_ce_accu.stride(0),
            student_ce_accu.stride(1),
            teacher_kl_max.stride(0),
            teacher_kl_max.stride(1),
            teacher_kl_accu.stride(0),
            teacher_kl_accu.stride(1),
            teacher_student_sum.stride(0),
            teacher_student_sum.stride(1),
            teacher_teacher_sum.stride(0),
            teacher_teacher_sum.stride(1),
        )

        student_ce_row_max, student_ce_log_denom = _reduce_lse_stats(student_ce_max, student_ce_accu)
        student_ce_lse = student_ce_row_max + student_ce_log_denom

        (
            teacher_kl_row_max,
            teacher_kl_log_denom,
            teacher_expect_student_local,
            teacher_expect_teacher_local,
        ) = _reduce_teacher_stats(
            teacher_kl_max,
            teacher_kl_accu,
            teacher_student_sum,
            teacher_teacher_sum,
        )

        teacher_kl_lse = teacher_kl_row_max + teacher_kl_log_denom
        local_teacher_kl_lse = teacher_kl_lse

        if vocab_parallel_options is not None:
            pg = vocab_parallel_options.group
            student_ce_lse = vp_reduce_lse(student_ce_lse, pg)
            target_logit = vp_reduce_correct_logit(target_logit, pg, dtype=student_ce_lse.dtype)

            teacher_kl_lse = vp_reduce_lse(teacher_kl_lse, pg)
            teacher_expect_student = _vp_reduce_teacher_expectation(
                teacher_expect_student_local,
                local_teacher_kl_lse,
                teacher_kl_lse,
                pg,
            )
            teacher_expect_teacher = _vp_reduce_teacher_expectation(
                teacher_expect_teacher_local,
                local_teacher_kl_lse,
                teacher_kl_lse,
                pg,
            )
        else:
            teacher_expect_student = teacher_expect_student_local
            teacher_expect_teacher = teacher_expect_teacher_local

        student_ce_loss = student_ce_lse - target_logit
        kl_loss = (
            student_ce_lse
            - teacher_kl_lse
            + teacher_expect_teacher
            - teacher_expect_student
        )
        all_loss = student_ce_loss + alpha * kl_loss

        ctx.alpha = float(alpha)
        ctx.chunk_size = int(chunk_size)
        ctx.vocab_start = int(vocab_start)
        ctx.save_for_backward(
            student_h,
            student_c,
            targets,
            teacher_h.detach(),
            teacher_c.detach(),
            student_ce_lse,
            teacher_kl_lse,
        )
        return all_loss, student_ce_loss, kl_loss

    @staticmethod
    def backward(
        ctx,
        grad_all_loss: torch.Tensor | None,
        grad_ce_loss: torch.Tensor | None,
        grad_kl_loss: torch.Tensor | None,
    ):
        (
            student_h,
            student_c,
            targets,
            teacher_h,
            teacher_c,
            student_ce_lse,
            teacher_kl_lse,
        ) = ctx.saved_tensors

        need_student_h = ctx.needs_input_grad[0]
        need_student_c = ctx.needs_input_grad[1]
        if not need_student_h and not need_student_c:
            return None, None, None, None, None, None, None, None

        num_rows = targets.numel()
        device = student_h.device

        if grad_all_loss is None:
            grad_all_loss = torch.zeros((num_rows,), device=device, dtype=torch.float32)
        else:
            grad_all_loss = grad_all_loss.reshape(-1).contiguous().float()

        if grad_ce_loss is None:
            grad_ce_loss = torch.zeros((num_rows,), device=device, dtype=torch.float32)
        else:
            grad_ce_loss = grad_ce_loss.reshape(-1).contiguous().float()

        if grad_kl_loss is None:
            grad_kl_loss = torch.zeros((num_rows,), device=device, dtype=torch.float32)
        else:
            grad_kl_loss = grad_kl_loss.reshape(-1).contiguous().float()

        ce_coeff = grad_all_loss + grad_ce_loss
        kl_coeff = grad_all_loss * ctx.alpha + grad_kl_loss

        if ce_coeff.numel() == 1:
            ce_coeff = ce_coeff.expand(num_rows).contiguous()
        if kl_coeff.numel() == 1:
            kl_coeff = kl_coeff.expand(num_rows).contiguous()

        grad_student_h_out, grad_student_c_out = _linear_cross_entropy_kl_backward_launcher(
            ce_coeff,
            kl_coeff,
            student_h,
            student_c,
            targets,
            teacher_h,
            teacher_c,
            student_ce_lse,
            teacher_kl_lse,
            vocab_start=ctx.vocab_start,
            need_student_h=need_student_h,
            need_student_c=need_student_c,
        )
        return grad_student_h_out, grad_student_c_out, None, None, None, None, None, None


def _zero_like_reduced(
    batch_shape: torch.Size,
    reference: torch.Tensor,
    reduction: str,
    *zero_refs: torch.Tensor,
) -> torch.Tensor:
    if not zero_refs:
        zero_refs = (reference,)

    zero_anchor = torch.zeros((), device=reference.device, dtype=reference.dtype)
    for zero_ref in zero_refs:
        zero_anchor = zero_anchor + zero_ref.sum() * 0.0

    if reduction == "none":
        return reference.new_zeros(batch_shape, dtype=torch.float32) + zero_anchor
    if reduction in {"sum", "mean"}:
        return reference.new_zeros((), dtype=torch.float32) + zero_anchor
    raise ValueError(f"Unknown reduction {reduction}")


def _reduce_active_loss(
    active_loss: torch.Tensor,
    batch_shape: torch.Size,
    valids: torch.Tensor | None,
    reduction: str,
) -> torch.Tensor:
    if reduction == "none":
        return handle_reduction_none(batch_shape, valids, 0, active_loss)
    if reduction == "sum":
        return active_loss.sum()
    if reduction == "mean":
        return active_loss.mean()
    raise ValueError(f"Unknown reduction {reduction}")


def _dense_linear_cross_entropy_kl(
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_logits = student_h @ student_c.t()
    teacher_logits = teacher_h.detach() @ teacher_c.detach().t()

    student_ce_loss = F.cross_entropy(student_logits.float(), targets, reduction="none")
    teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    teacher_prob = teacher_log_prob.exp()
    kl_loss = torch.sum(teacher_prob * (teacher_log_prob - student_log_prob), dim=-1)
    all_loss = student_ce_loss + alpha * kl_loss
    return all_loss, student_ce_loss, kl_loss


@overload
def linear_cross_entropy_kl(
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    *,
    alpha: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
    return_components: Literal[False] = False,
    chunk_size: int = 16384,
    vocab_parallel_options: VocabParallelOptions | None = None,
) -> torch.Tensor: ...


@overload
def linear_cross_entropy_kl(
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    *,
    alpha: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
    return_components: Literal[True],
    chunk_size: int = 16384,
    vocab_parallel_options: VocabParallelOptions | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def linear_cross_entropy_kl(
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    *,
    alpha: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
    return_components: bool = False,
    chunk_size: int = 16384,
    vocab_parallel_options: VocabParallelOptions | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes fused student CE + alpha * forward KL distillation.

    See `_LINEAR_CROSS_ENTROPY_KL_DOC` in this module for the full semantics.
    """
    _validate_inputs(
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        vocab_parallel_options,
    )

    student_h = student_h.contiguous()
    teacher_h = teacher_h.contiguous()
    student_c = student_c.contiguous()
    teacher_c = teacher_c.contiguous()
    targets = targets.contiguous()

    batch_shape = targets.shape
    student_h_flat = student_h.flatten(0, -2)
    teacher_h_flat = teacher_h.flatten(0, -2)
    targets_flat = targets.flatten()

    if targets_flat.numel() == 0:
        raise ValueError("targets must contain at least one element")

    valids = (targets_flat != ignore_index).nonzero(as_tuple=False).squeeze(1)
    if valids.numel() == targets_flat.numel():
        active_student_h = student_h_flat
        active_teacher_h = teacher_h_flat
        active_targets = targets_flat
        valids_for_scatter = None
    else:
        active_student_h = student_h_flat.index_select(0, valids).contiguous()
        active_teacher_h = teacher_h_flat.index_select(0, valids).contiguous()
        active_targets = targets_flat.index_select(0, valids).contiguous()
        valids_for_scatter = valids

    if active_targets.numel() == 0:
        zero_all = _zero_like_reduced(batch_shape, student_h, reduction, student_h, student_c)
        if return_components:
            zero_ce = _zero_like_reduced(batch_shape, student_h, reduction, student_h, student_c)
            zero_kl = _zero_like_reduced(batch_shape, student_h, reduction, student_h, student_c)
            return zero_all, zero_ce, zero_kl
        return zero_all

    if torch.any(active_targets < 0):
        raise ValueError("targets must be non-negative or equal to ignore_index")

    if vocab_parallel_options is None:
        if torch.any(active_targets >= student_c.size(0)):
            raise ValueError("targets must be in [0, vocab_size) or equal to ignore_index")
    else:
        active_student_h = vp_reduce_e_grad_hook(active_student_h, vocab_parallel_options)

    active_all_loss, active_ce_loss, active_kl_loss = _LinearCrossEntropyKLFunction.apply(
        active_student_h,
        student_c,
        active_targets,
        active_teacher_h,
        teacher_c,
        float(alpha),
        int(chunk_size),
        vocab_parallel_options,
    )

    all_loss = _reduce_active_loss(active_all_loss, batch_shape, valids_for_scatter, reduction)
    if not return_components:
        return all_loss

    ce_loss = _reduce_active_loss(active_ce_loss, batch_shape, valids_for_scatter, reduction)
    kl_loss = _reduce_active_loss(active_kl_loss, batch_shape, valids_for_scatter, reduction)
    return all_loss, ce_loss, kl_loss


class LinearCrossEntropyKL(nn.Module):
    def __init__(
        self,
        *,
        alpha: float = 1.0,
        ignore_index: int = IGNORE_INDEX,
        reduction: str = "mean",
        return_components: bool = False,
        chunk_size: int = 16384,
        vocab_parallel_options: VocabParallelOptions | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.return_components = return_components
        self.chunk_size = chunk_size
        self.vocab_parallel_options = vocab_parallel_options

    def forward(
        self,
        student_h: torch.Tensor,
        student_c: torch.Tensor,
        targets: torch.Tensor,
        teacher_h: torch.Tensor,
        teacher_c: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return linear_cross_entropy_kl(
            student_h,
            student_c,
            targets,
            teacher_h,
            teacher_c,
            alpha=self.alpha,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            return_components=self.return_components,
            chunk_size=self.chunk_size,
            vocab_parallel_options=self.vocab_parallel_options,
        )