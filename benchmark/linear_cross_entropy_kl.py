# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
from fire import Fire

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cut_cross_entropy import linear_cross_entropy_kl
from cut_cross_entropy.linear_cross_entropy_kl import (
    _LinearCrossEntropyKLFunction,
    _dense_linear_cross_entropy_kl,
    _linear_cross_entropy_kl_backward_kernel,
)
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.tl_utils import b_bin_fn, is_triton_greater_or_equal_3_2_0


MIB = 1024**2


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    batch_size: int
    seq_len: int
    vocab_size: int
    hidden_dim: int


@dataclass(frozen=True)
class BenchmarkResult:
    status: str
    forward_ms: float | None = None
    backward_ms: float | None = None
    total_ms: float | None = None
    forward_peak_mib: float | None = None
    total_peak_mib: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class BackwardBreakdownResult:
    num_rows: int
    common_ms: float
    de_total_ms: float
    dc_total_ms: float
    full_ms: float
    de_extra_ms: float
    dc_extra_ms: float
    interaction_ms: float


@dataclass(frozen=True)
class AccuracyResult:
    all_max_abs: float
    ce_max_abs: float
    kl_max_abs: float
    grad_h_max_abs: float
    grad_c_max_abs: float
    all_max_rel: float
    ce_max_rel: float
    kl_max_rel: float
    grad_h_max_rel: float
    grad_c_max_rel: float


class _ProfileCtx:
    saved_tensors: tuple[torch.Tensor, ...]

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        self.saved_tensors = tensors


DEFAULT_CASES = [
    BenchmarkCase("small-s64-v32k-d2048", 1, 64, 32768, 2048),
    BenchmarkCase("small-s256-v32k-d2048", 1, 256, 32768, 2048),
    BenchmarkCase("mid-b4-s256-v32k-d2048", 4, 256, 32768, 2048),
    BenchmarkCase("long-s8192-v4k-d2048", 1, 8192, 4096, 2048),
    BenchmarkCase("long-s32768-v4k-d2048", 1, 32768, 4096, 2048),
    BenchmarkCase("long-s65536-v4k-d2048", 1, 65536, 4096, 2048),
    BenchmarkCase("long-s65536-v8k-d2048", 1, 65536, 8192, 2048),
]

LONG_SEQ_CASES = [
    BenchmarkCase("long-s8192-v4k-d2048", 1, 8192, 4096, 2048),
    BenchmarkCase("long-s16384-v4k-d2048", 1, 16384, 4096, 2048),
    BenchmarkCase("long-s32768-v4k-d2048", 1, 32768, 4096, 2048),
    BenchmarkCase("long-s65536-v4k-d2048", 1, 65536, 4096, 2048),
    BenchmarkCase("long-s65536-v8k-d2048", 1, 65536, 8192, 2048),
]

BATCH_SIZE_CASES = [
    BenchmarkCase("batch-b1-s8192-v4k-d2048", 1, 8192, 4096, 2048),
    BenchmarkCase("batch-b2-s8192-v4k-d2048", 2, 8192, 4096, 2048),
    BenchmarkCase("batch-b4-s8192-v4k-d2048", 4, 8192, 4096, 2048),
    BenchmarkCase("batch-b8-s8192-v4k-d2048", 8, 8192, 4096, 2048),
]

HIDDEN_DIM_CASES = [
    BenchmarkCase("hidden-s8192-v4k-d1024", 1, 8192, 4096, 1024),
    BenchmarkCase("hidden-s8192-v4k-d2048", 1, 8192, 4096, 2048),
    BenchmarkCase("hidden-s8192-v4k-d4096", 1, 8192, 4096, 4096),
]

BACKWARD_BREAKDOWN_CASES = [
    BenchmarkCase("long-s8192-v4k-d2048", 1, 8192, 4096, 2048),
    BenchmarkCase("long-s65536-v4k-d2048", 1, 65536, 4096, 2048),
    BenchmarkCase("long-s65536-v8k-d2048", 1, 65536, 8192, 2048),
]

ACCURACY_CASES = [
    BenchmarkCase("acc-small-s256-v32k-d2048", 1, 256, 32768, 2048),
    BenchmarkCase("acc-mid-b4-s256-v32k-d2048", 4, 256, 32768, 2048),
    BenchmarkCase("acc-long-s8192-v4k-d2048", 1, 8192, 4096, 2048),
]


def _zero_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad = None


def _build_case_tensors(
    case: BenchmarkCase,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_scale = case.hidden_dim**0.5
    student_h = torch.randn(
        (case.batch_size, case.seq_len, case.hidden_dim),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    ) / hidden_scale
    teacher_h = torch.randn(
        (case.batch_size, case.seq_len, case.hidden_dim),
        device="cuda",
        dtype=dtype,
    ) / hidden_scale
    student_c = torch.randn(
        (case.vocab_size, case.hidden_dim),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    teacher_c = torch.randn(
        (case.vocab_size, case.hidden_dim),
        device="cuda",
        dtype=dtype,
    )
    targets = torch.randint(0, case.vocab_size, size=(case.batch_size, case.seq_len), device="cuda")
    targets = torch.where(
        torch.rand((case.batch_size, case.seq_len), device="cuda") < 0.05,
        torch.full_like(targets, IGNORE_INDEX),
        targets,
    )
    return student_h, student_c, targets, teacher_h, teacher_c


def _active_case_tensors(
    case: BenchmarkCase,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    student_h, student_c, targets, teacher_h, teacher_c = _build_case_tensors(case, dtype=dtype)
    targets_flat = targets.flatten()
    active = targets_flat != IGNORE_INDEX
    student_h_flat = student_h.flatten(0, -2)
    teacher_h_flat = teacher_h.flatten(0, -2)
    if torch.all(active):
        return (
            student_h_flat.contiguous(),
            student_c,
            targets_flat.contiguous(),
            teacher_h_flat.contiguous(),
            teacher_c,
        )
    return (
        student_h_flat[active].contiguous(),
        student_c,
        targets_flat[active].contiguous(),
        teacher_h_flat[active].contiguous(),
        teacher_c,
    )


def _time_cuda_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(max(warmup, 0)):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(max(iters, 1)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def _prepare_backward_profile_state(
    case: BenchmarkCase,
    *,
    dtype: torch.dtype,
    alpha: float,
    chunk_size: int,
) -> dict[str, object]:
    student_h, student_c, targets, teacher_h, teacher_c = _active_case_tensors(case, dtype=dtype)

    ctx = _ProfileCtx()
    _LinearCrossEntropyKLFunction.forward(
        ctx,
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        alpha,
        chunk_size,
        None,
    )

    (
        saved_student_h,
        saved_student_c,
        saved_targets,
        saved_teacher_h,
        saved_teacher_c,
        student_ce_lse,
        teacher_kl_lse,
    ) = ctx.saved_tensors
    num_rows = saved_targets.numel()
    if num_rows <= 0:
        raise ValueError("Backward profiling requires at least one active target")

    can_use_fp32_accum = is_triton_greater_or_equal_3_2_0()
    use_kahan_e = not can_use_fp32_accum
    use_kahan_c = not can_use_fp32_accum
    de_dtype = torch.float32 if can_use_fp32_accum else saved_student_h.dtype
    dc_dtype = torch.float32 if can_use_fp32_accum else saved_student_c.dtype

    n_d_locks = triton.cdiv(saved_student_c.size(1), 64)
    dE = torch.empty_like(saved_student_h, dtype=de_dtype)
    dC = torch.empty_like(saved_student_c, dtype=dc_dtype)
    dEC = torch.empty_like(saved_student_h) if use_kahan_e else saved_student_h.new_empty((1, 1))
    dCC = torch.empty_like(saved_student_c) if use_kahan_c else saved_student_c.new_empty((1, 1))
    dELocks = saved_student_h.new_empty((triton.cdiv(saved_student_h.size(0), 128), n_d_locks), dtype=torch.int32)
    dCLocks = saved_student_c.new_empty((triton.cdiv(saved_student_c.size(0), 128), n_d_locks), dtype=torch.int32)
    dummy_grad = saved_student_h.new_empty((1, 1))
    dummy_locks = saved_student_h.new_empty((1, 1), dtype=torch.int32)

    ce_coeff = torch.full((num_rows,), 1.0 / num_rows, device=saved_student_h.device, dtype=torch.float32)
    kl_coeff = ce_coeff * alpha

    def grid(meta):
        return (
            triton.cdiv(saved_student_h.size(0), meta["BLOCK_B"])
            * triton.cdiv(saved_student_c.size(0), meta["BLOCK_V"]),
        )

    return {
        "student_h": saved_student_h,
        "student_c": saved_student_c,
        "teacher_h": saved_teacher_h,
        "teacher_c": saved_teacher_c,
        "targets": saved_targets,
        "student_ce_lse": student_ce_lse,
        "teacher_kl_lse": teacher_kl_lse,
        "ce_coeff": ce_coeff,
        "kl_coeff": kl_coeff,
        "dE": dE,
        "dC": dC,
        "dEC": dEC,
        "dCC": dCC,
        "dELocks": dELocks,
        "dCLocks": dCLocks,
        "dummy_grad": dummy_grad,
        "dummy_locks": dummy_locks,
        "use_kahan_e": use_kahan_e,
        "use_kahan_c": use_kahan_c,
        "grid": grid,
        "vocab_start": ctx.vocab_start,
        "num_rows": num_rows,
        "b_bin": b_bin_fn(saved_student_h.size(0)),
    }


def _zero_backward_profile_state(state: dict[str, object], *, use_de: bool, use_dc: bool) -> None:
    if use_de:
        state["dE"].zero_()
        state["dELocks"].zero_()
        if state["use_kahan_e"]:
            state["dEC"].zero_()
    if use_dc:
        state["dC"].zero_()
        state["dCLocks"].zero_()
        if state["use_kahan_c"]:
            state["dCC"].zero_()


def _launch_profile_backward_kernel(state: dict[str, object], *, use_de: bool, use_dc: bool) -> None:
    _linear_cross_entropy_kl_backward_kernel[state["grid"]](
        state["student_h"],
        state["student_c"],
        state["teacher_h"],
        state["teacher_c"],
        state["student_ce_lse"],
        state["teacher_kl_lse"],
        state["ce_coeff"],
        state["kl_coeff"],
        state["targets"],
        state["dE"] if use_de else state["dummy_grad"],
        state["dEC"] if (use_de and state["use_kahan_e"]) else state["dummy_grad"],
        state["dELocks"] if use_de else state["dummy_locks"],
        state["dC"] if use_dc else state["dummy_grad"],
        state["dCC"] if (use_dc and state["use_kahan_c"]) else state["dummy_grad"],
        state["dCLocks"] if use_dc else state["dummy_locks"],
        state["student_h"].size(0),
        state["student_h"].size(1),
        state["student_c"].size(0),
        state["dELocks"].size(0) if use_de else state["dummy_locks"].size(0),
        state["dELocks"].size(1) if use_de else state["dummy_locks"].size(1),
        state["dCLocks"].size(0) if use_dc else state["dummy_locks"].size(0),
        state["dCLocks"].size(1) if use_dc else state["dummy_locks"].size(1),
        state["student_h"].stride(0),
        state["student_h"].stride(1),
        state["student_c"].stride(0),
        state["student_c"].stride(1),
        state["teacher_h"].stride(0),
        state["teacher_h"].stride(1),
        state["teacher_c"].stride(0),
        state["teacher_c"].stride(1),
        vocab_start=state["vocab_start"],
        USE_KAHAN_E=bool(state["use_kahan_e"] and use_de),
        USE_KAHAN_C=bool(state["use_kahan_c"] and use_dc),
        USE_DE=use_de,
        USE_DC=use_dc,
        B_BIN=state["b_bin"],
    )


def _profile_backward_breakdown_case(
    case: BenchmarkCase,
    *,
    dtype: torch.dtype,
    alpha: float,
    chunk_size: int,
    warmup: int,
    iters: int,
) -> BackwardBreakdownResult:
    state = _prepare_backward_profile_state(case, dtype=dtype, alpha=alpha, chunk_size=chunk_size)

    def time_variant(*, use_de: bool, use_dc: bool) -> float:
        for _ in range(max(warmup, 0)):
            _zero_backward_profile_state(state, use_de=use_de, use_dc=use_dc)
            _launch_profile_backward_kernel(state, use_de=use_de, use_dc=use_dc)
        torch.cuda.synchronize()

        times: list[float] = []
        for _ in range(max(iters, 1)):
            _zero_backward_profile_state(state, use_de=use_de, use_dc=use_dc)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _launch_profile_backward_kernel(state, use_de=use_de, use_dc=use_dc)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        return sum(times) / len(times)

    common_ms = time_variant(use_de=False, use_dc=False)
    de_total_ms = time_variant(use_de=True, use_dc=False)
    dc_total_ms = time_variant(use_de=False, use_dc=True)
    full_ms = time_variant(use_de=True, use_dc=True)

    de_extra_ms = de_total_ms - common_ms
    dc_extra_ms = dc_total_ms - common_ms
    interaction_ms = full_ms - common_ms - de_extra_ms - dc_extra_ms

    return BackwardBreakdownResult(
        num_rows=int(state["num_rows"]),
        common_ms=common_ms,
        de_total_ms=de_total_ms,
        dc_total_ms=dc_total_ms,
        full_ms=full_ms,
        de_extra_ms=de_extra_ms,
        dc_extra_ms=dc_extra_ms,
        interaction_ms=interaction_ms,
    )


def _print_backward_breakdown(case: BenchmarkCase, result: BackwardBreakdownResult) -> None:
    print(f"case: {case.name} | B={case.batch_size} S={case.seq_len} V={case.vocab_size} D={case.hidden_dim}")
    print(f"active_rows: {result.num_rows}")
    print(
        "backward-kernel: "
        f"common(softmax+LSE+prob) {result.common_ms:.2f} ms | "
        f"dE_matmul_extra {result.de_extra_ms:.2f} ms | "
        f"dC_accum_extra {result.dc_extra_ms:.2f} ms | "
        f"residual {result.interaction_ms:+.2f} ms | "
        f"full {result.full_ms:.2f} ms"
    )
    if result.full_ms > 0:
        print(
            "shares: "
            f"common {100.0 * result.common_ms / result.full_ms:.1f}% | "
            f"dE {100.0 * result.de_extra_ms / result.full_ms:.1f}% | "
            f"dC {100.0 * result.dc_extra_ms / result.full_ms:.1f}% | "
            f"residual {100.0 * result.interaction_ms / result.full_ms:.1f}%"
        )


def _max_rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    significant = b_f.abs() > 1e-4
    if not torch.any(significant):
        return 0.0
    diff = (a_f[significant] - b_f[significant]).abs()
    denom = b_f[significant].abs()
    return (diff / denom).max().item()


def _exact_linear_cross_entropy_kl(
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_logits = student_h.float() @ student_c.float().t()
    teacher_logits = teacher_h.detach().float() @ teacher_c.detach().float().t()

    student_ce_loss = torch.nn.functional.cross_entropy(student_logits, targets, reduction="none")
    teacher_log_prob = torch.nn.functional.log_softmax(teacher_logits, dim=-1)
    student_log_prob = torch.nn.functional.log_softmax(student_logits, dim=-1)
    teacher_prob = teacher_log_prob.exp()
    kl_loss = torch.sum(teacher_prob * (teacher_log_prob - student_log_prob), dim=-1)
    all_loss = student_ce_loss + alpha * kl_loss
    return all_loss, student_ce_loss, kl_loss


def _accuracy_case(
    case: BenchmarkCase,
    *,
    dtype: torch.dtype,
    alpha: float,
    reference: str,
) -> AccuracyResult:
    student_h, student_c, targets, teacher_h, teacher_c = _build_case_tensors(case, dtype=dtype)
    total_rows = targets.numel()
    targets_flat = targets.flatten()
    active = targets_flat != IGNORE_INDEX

    if reference == "dense":
        ref_student_h = student_h.detach().clone().requires_grad_(True)
        ref_student_c = student_c.detach().clone().requires_grad_(True)
        ref_teacher_h = teacher_h.detach()
        ref_teacher_c = teacher_c.detach()
        reference_impl = _dense_linear_cross_entropy_kl
    elif reference == "exact":
        ref_student_h = student_h.detach().float().clone().requires_grad_(True)
        ref_student_c = student_c.detach().float().clone().requires_grad_(True)
        ref_teacher_h = teacher_h.detach().float()
        ref_teacher_c = teacher_c.detach().float()
        reference_impl = _exact_linear_cross_entropy_kl
    else:
        raise ValueError(f"Unknown reference {reference}")

    ref_student_h_flat = ref_student_h.flatten(0, -2)
    ref_teacher_h_flat = ref_teacher_h.flatten(0, -2)
    active_student_h = ref_student_h_flat[active].contiguous()
    active_teacher_h = ref_teacher_h_flat[active].contiguous()
    active_targets = targets_flat[active].contiguous()

    ref_all_active, ref_ce_active, ref_kl_active = reference_impl(
        active_student_h,
        ref_student_c,
        active_targets,
        active_teacher_h,
        ref_teacher_c,
        alpha=alpha,
    )

    ref_all = torch.zeros_like(targets_flat, dtype=torch.float32)
    ref_ce = torch.zeros_like(targets_flat, dtype=torch.float32)
    ref_kl = torch.zeros_like(targets_flat, dtype=torch.float32)
    ref_all[active] = ref_all_active
    ref_ce[active] = ref_ce_active
    ref_kl[active] = ref_kl_active

    ref_loss = ref_all_active.sum() / total_rows
    ref_loss.backward()

    fused_student_h = student_h.detach().clone().requires_grad_(True)
    fused_student_c = student_c.detach().clone().requires_grad_(True)
    fused_all, fused_ce, fused_kl = linear_cross_entropy_kl(
        fused_student_h,
        fused_student_c,
        targets,
        teacher_h,
        teacher_c,
        alpha=alpha,
        reduction="none",
        return_components=True,
    )
    fused_loss = fused_all.mean()
    fused_loss.backward()

    fused_all_flat = fused_all.flatten()
    fused_ce_flat = fused_ce.flatten()
    fused_kl_flat = fused_kl.flatten()

    assert ref_student_h.grad is not None
    assert ref_student_c.grad is not None
    assert fused_student_h.grad is not None
    assert fused_student_c.grad is not None

    return AccuracyResult(
        all_max_abs=(fused_all_flat.float() - ref_all).abs().max().item(),
        ce_max_abs=(fused_ce_flat.float() - ref_ce).abs().max().item(),
        kl_max_abs=(fused_kl_flat.float() - ref_kl).abs().max().item(),
        grad_h_max_abs=(fused_student_h.grad.float() - ref_student_h.grad.float()).abs().max().item(),
        grad_c_max_abs=(fused_student_c.grad.float() - ref_student_c.grad.float()).abs().max().item(),
        all_max_rel=_max_rel_diff(fused_all_flat, ref_all),
        ce_max_rel=_max_rel_diff(fused_ce_flat, ref_ce),
        kl_max_rel=_max_rel_diff(fused_kl_flat, ref_kl),
        grad_h_max_rel=_max_rel_diff(fused_student_h.grad, ref_student_h.grad),
        grad_c_max_rel=_max_rel_diff(fused_student_c.grad, ref_student_c.grad),
    )


def _print_accuracy_result(case: BenchmarkCase, dtype: str, reference: str, result: AccuracyResult) -> None:
    print(
        f"case: {case.name} | dtype={dtype} | reference={reference} | "
        f"B={case.batch_size} S={case.seq_len} V={case.vocab_size} D={case.hidden_dim}"
    )
    print(
        "forward_max_abs: "
        f"all {result.all_max_abs:.6e} | "
        f"ce {result.ce_max_abs:.6e} | "
        f"kl {result.kl_max_abs:.6e}"
    )
    print(
        "backward_max_abs: "
        f"grad_h {result.grad_h_max_abs:.6e} | "
        f"grad_c {result.grad_c_max_abs:.6e}"
    )
    print(
        "forward_max_rel: "
        f"all {result.all_max_rel:.6e} | "
        f"ce {result.ce_max_rel:.6e} | "
        f"kl {result.kl_max_rel:.6e}"
    )
    print(
        "backward_max_rel: "
        f"grad_h {result.grad_h_max_rel:.6e} | "
        f"grad_c {result.grad_c_max_rel:.6e}"
    )


def _dense_loss(
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    all_loss, _, _ = _dense_linear_cross_entropy_kl(
        student_h.flatten(0, -2),
        student_c,
        targets.flatten(),
        teacher_h.flatten(0, -2),
        teacher_c,
        alpha=alpha,
    )
    return all_loss.view_as(targets)


def _fused_loss(
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    return linear_cross_entropy_kl(
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        alpha=alpha,
        reduction="none",
    )


def _measure_impl(
    label: str,
    fn,
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    *,
    warmup: int,
    backward: bool,
) -> BenchmarkResult:
    try:
        for _ in range(max(warmup, 0)):
            _zero_grads(student_h, student_c)
            out = fn(student_h, student_c, targets, teacher_h, teacher_c)
            if backward:
                out.mean().backward()
            torch.cuda.synchronize()

        _zero_grads(student_h, student_c)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_forward = time.perf_counter()
        out = fn(student_h, student_c, targets, teacher_h, teacher_c)
        torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - start_forward) * 1000.0
        forward_peak_mib = torch.cuda.max_memory_allocated() / MIB

        if backward:
            start_backward = time.perf_counter()
            out.mean().backward()
            torch.cuda.synchronize()
            backward_ms = (time.perf_counter() - start_backward) * 1000.0
        else:
            backward_ms = 0.0

        total_peak_mib = torch.cuda.max_memory_allocated() / MIB
        total_ms = forward_ms + backward_ms
        _zero_grads(student_h, student_c)

        print(
            f"{label}: "
            f"fwd {forward_ms:.2f} ms | "
            f"bwd {backward_ms:.2f} ms | "
            f"total {total_ms:.2f} ms | "
            f"fwd_peak {forward_peak_mib:.1f} MiB | "
            f"total_peak {total_peak_mib:.1f} MiB"
        )
        return BenchmarkResult(
            status="ok",
            forward_ms=forward_ms,
            backward_ms=backward_ms,
            total_ms=total_ms,
            forward_peak_mib=forward_peak_mib,
            total_peak_mib=total_peak_mib,
        )
    except RuntimeError as exc:
        message = str(exc)
        _zero_grads(student_h, student_c)
        torch.cuda.empty_cache()
        if "out of memory" not in message.lower():
            raise
        print(f"{label}: OOM ({message.splitlines()[0]})")
        return BenchmarkResult(status="oom", error=message.splitlines()[0])


def _run_case(
    case: BenchmarkCase,
    *,
    dtype: torch.dtype,
    alpha: float,
    warmup: int,
    backward: bool,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    student_h, student_c, targets, teacher_h, teacher_c = _build_case_tensors(case, dtype=dtype)

    dense_result = _measure_impl(
        "dense",
        lambda sh, sc, tg, th, tc: _dense_loss(
            sh,
            sc,
            tg,
            th,
            tc,
            alpha=alpha,
        ),
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        warmup=warmup,
        backward=backward,
    )

    fused_result = _measure_impl(
        "fused",
        lambda sh, sc, tg, th, tc: _fused_loss(
            sh,
            sc,
            tg,
            th,
            tc,
            alpha=alpha,
        ),
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        warmup=warmup,
        backward=backward,
    )

    return dense_result, fused_result


def _print_case_delta(dense_result: BenchmarkResult, fused_result: BenchmarkResult) -> None:
    if dense_result.status != "ok" or fused_result.status != "ok":
        return

    assert dense_result.total_ms is not None
    assert fused_result.total_ms is not None
    assert dense_result.total_peak_mib is not None
    assert fused_result.total_peak_mib is not None
    assert dense_result.forward_ms is not None
    assert fused_result.forward_ms is not None
    assert dense_result.backward_ms is not None
    assert fused_result.backward_ms is not None

    print(
        "delta: "
        f"fwd {fused_result.forward_ms - dense_result.forward_ms:+.2f} ms | "
        f"bwd {fused_result.backward_ms - dense_result.backward_ms:+.2f} ms | "
        f"total {fused_result.total_ms - dense_result.total_ms:+.2f} ms | "
        f"peak {fused_result.total_peak_mib - dense_result.total_peak_mib:+.1f} MiB"
    )


def benchmark(
    batch_size: int = 4,
    seq_len: int = 256,
    vocab_size: int = 32768,
    hidden_dim: int = 2048,
    dtype: str = "bfloat16",
    alpha: float = 1.0,
    warmup: int = 1,
    backward: bool = True,
) -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    case = BenchmarkCase("single", batch_size, seq_len, vocab_size, hidden_dim)
    print(
        f"case: B={case.batch_size} S={case.seq_len} V={case.vocab_size} D={case.hidden_dim} dtype={dtype}"
    )
    dense_result, fused_result = _run_case(
        case,
        dtype=getattr(torch, dtype),
        alpha=alpha,
        warmup=warmup,
        backward=backward,
    )
    _print_case_delta(dense_result, fused_result)


def sweep(
    preset: str = "default",
    dtype: str = "bfloat16",
    alpha: float = 1.0,
    warmup: int = 1,
    backward: bool = True,
) -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    if preset == "default":
        cases = DEFAULT_CASES
    elif preset == "long-seq":
        cases = LONG_SEQ_CASES
    elif preset == "batch-size":
        cases = BATCH_SIZE_CASES
    elif preset == "hidden-dim":
        cases = HIDDEN_DIM_CASES
    else:
        raise ValueError(f"Unknown preset {preset}")

    dtype_obj = getattr(torch, dtype)
    for case in cases:
        print()
        print(f"case: {case.name} | B={case.batch_size} S={case.seq_len} V={case.vocab_size} D={case.hidden_dim}")
        dense_result, fused_result = _run_case(
            case,
            dtype=dtype_obj,
            alpha=alpha,
            warmup=warmup,
            backward=backward,
        )
        _print_case_delta(dense_result, fused_result)


def profile_backward(
    batch_size: int = 1,
    seq_len: int = 65536,
    vocab_size: int = 4096,
    hidden_dim: int = 2048,
    dtype: str = "bfloat16",
    alpha: float = 1.0,
    chunk_size: int = 16384,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    case = BenchmarkCase("single", batch_size, seq_len, vocab_size, hidden_dim)
    result = _profile_backward_breakdown_case(
        case,
        dtype=getattr(torch, dtype),
        alpha=alpha,
        chunk_size=chunk_size,
        warmup=warmup,
        iters=iters,
    )
    _print_backward_breakdown(case, result)


def profile_backward_sweep(
    preset: str = "backward-breakdown",
    dtype: str = "bfloat16",
    alpha: float = 1.0,
    chunk_size: int = 16384,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    if preset == "backward-breakdown":
        cases = BACKWARD_BREAKDOWN_CASES
    elif preset == "batch-size":
        cases = BATCH_SIZE_CASES
    elif preset == "hidden-dim":
        cases = HIDDEN_DIM_CASES
    else:
        raise ValueError(f"Unknown preset {preset}")

    dtype_obj = getattr(torch, dtype)
    for case in cases:
        print()
        result = _profile_backward_breakdown_case(
            case,
            dtype=dtype_obj,
            alpha=alpha,
            chunk_size=chunk_size,
            warmup=warmup,
            iters=iters,
        )
        _print_backward_breakdown(case, result)


def accuracy(
    batch_size: int = 1,
    seq_len: int = 256,
    vocab_size: int = 32768,
    hidden_dim: int = 2048,
    dtype: str = "bfloat16",
    alpha: float = 1.0,
    reference: str = "dense",
) -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("highest")

    case = BenchmarkCase("single", batch_size, seq_len, vocab_size, hidden_dim)
    result = _accuracy_case(case, dtype=getattr(torch, dtype), alpha=alpha, reference=reference)
    _print_accuracy_result(case, dtype, reference, result)


def accuracy_sweep(
    preset: str = "default",
    dtype: str = "bfloat16",
    alpha: float = 1.0,
    reference: str = "dense",
) -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("highest")

    if preset == "default":
        cases = ACCURACY_CASES
    else:
        raise ValueError(f"Unknown preset {preset}")

    dtype_obj = getattr(torch, dtype)
    for case in cases:
        print()
        result = _accuracy_case(case, dtype=dtype_obj, alpha=alpha, reference=reference)
        _print_accuracy_result(case, dtype, reference, result)


if __name__ == "__main__":
    Fire(
        {
            "benchmark": benchmark,
            "sweep": sweep,
            "profile_backward": profile_backward,
            "profile_backward_sweep": profile_backward_sweep,
            "accuracy": accuracy,
            "accuracy_sweep": accuracy_sweep,
        }
    )