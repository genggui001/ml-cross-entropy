# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch
import torch.nn.functional as F

from cut_cross_entropy import linear_cross_entropy_kl
from cut_cross_entropy.constants import IGNORE_INDEX

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _reference_loss(
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    *,
    alpha: float,
    reduction: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_shape = targets.shape
    student_h_flat = student_h.flatten(0, -2).float()
    teacher_h_flat = teacher_h.flatten(0, -2).float()
    targets_flat = targets.flatten()
    active = targets_flat != IGNORE_INDEX

    if not torch.any(active):
        zero = student_h.new_zeros(batch_shape, dtype=torch.float32)
        if reduction == "none":
            return zero, zero, zero
        scalar_zero = student_h.new_zeros((), dtype=torch.float32)
        return scalar_zero, scalar_zero, scalar_zero

    active_student_h = student_h_flat[active]
    active_teacher_h = teacher_h_flat[active]
    active_targets = targets_flat[active]

    student_logits = active_student_h @ student_c.float().t()
    teacher_logits = active_teacher_h @ teacher_c.float().t()

    ce_loss = F.cross_entropy(student_logits, active_targets, reduction="none")
    teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
    student_log_prob = F.log_softmax(student_logits, dim=-1)
    teacher_prob = teacher_log_prob.exp()
    kl_loss = torch.sum(teacher_prob * (teacher_log_prob - student_log_prob), dim=-1)
    all_loss = ce_loss + alpha * kl_loss

    if reduction == "none":
        full_all = student_h.new_zeros((batch_shape.numel(),), dtype=torch.float32)
        full_ce = student_h.new_zeros((batch_shape.numel(),), dtype=torch.float32)
        full_kl = student_h.new_zeros((batch_shape.numel(),), dtype=torch.float32)
        full_all[active] = all_loss
        full_ce[active] = ce_loss
        full_kl[active] = kl_loss
        return (
            full_all.view(batch_shape),
            full_ce.view(batch_shape),
            full_kl.view(batch_shape),
        )
    if reduction == "sum":
        return all_loss.sum(), ce_loss.sum(), kl_loss.sum()
    if reduction == "mean":
        return all_loss.mean(), ce_loss.mean(), kl_loss.mean()
    raise ValueError(f"Unknown reduction {reduction}")


@skip_no_cuda
@pytest.mark.parametrize(
    "dtype,error_tol",
    [(torch.float32, 1e-5), (torch.float16, 3e-3), (torch.bfloat16, 2.5e-2)],
)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("invalids", [False, True])
@pytest.mark.parametrize("shape", [((4, 33), 257, 64), ((3, 17), 509, 96)])
def test_linear_cross_entropy_kl_forward(
    dtype: torch.dtype,
    error_tol: float,
    reduction: str,
    invalids: bool,
    shape: tuple[tuple[int, int], int, int],
):
    torch.cuda.manual_seed(0)

    batch_shape, vocab_size, hidden_dim = shape
    student_h = torch.randn((*batch_shape, hidden_dim), device="cuda", dtype=dtype) / (hidden_dim**0.5)
    teacher_h = torch.randn((*batch_shape, hidden_dim), device="cuda", dtype=dtype) / (hidden_dim**0.5)
    student_c = torch.randn((vocab_size, hidden_dim), device="cuda", dtype=dtype)
    teacher_c = torch.randn((vocab_size, hidden_dim), device="cuda", dtype=dtype)
    targets = torch.randint(0, vocab_size, size=batch_shape, device="cuda")

    if invalids:
        invalid_mask = torch.rand(batch_shape, device="cuda") < 0.2
        targets = torch.where(invalid_mask, torch.full_like(targets, IGNORE_INDEX), targets)

    alpha = 0.37
    ref_all, ref_ce, ref_kl = _reference_loss(
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        alpha=alpha,
        reduction=reduction,
    )
    all_loss, ce_loss, kl_loss = linear_cross_entropy_kl(
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        alpha=alpha,
        reduction=reduction,
        return_components=True,
    )

    assert torch.allclose(all_loss.float(), ref_all.float(), atol=error_tol, rtol=error_tol)
    assert torch.allclose(ce_loss.float(), ref_ce.float(), atol=error_tol, rtol=error_tol)
    assert torch.allclose(kl_loss.float(), ref_kl.float(), atol=error_tol, rtol=error_tol)
    assert torch.allclose(
        all_loss.float(),
        (ce_loss + alpha * kl_loss).float(),
        atol=error_tol,
        rtol=error_tol,
    )


def _reference_grads(
    student_h: torch.Tensor,
    student_c: torch.Tensor,
    targets: torch.Tensor,
    teacher_h: torch.Tensor,
    teacher_c: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_student_h = student_h.detach().clone().requires_grad_(True)
    ref_student_c = student_c.detach().clone().requires_grad_(True)
    ref_all, _, _ = _reference_loss(
        ref_student_h,
        ref_student_c,
        targets,
        teacher_h.detach(),
        teacher_c.detach(),
        alpha=alpha,
        reduction="mean",
    )
    ref_all.backward()
    assert ref_student_h.grad is not None
    assert ref_student_c.grad is not None
    return ref_student_h.grad.detach().clone(), ref_student_c.grad.detach().clone()


@skip_no_cuda
@pytest.mark.parametrize(
    "dtype,error_tol",
    [(torch.float32, 1e-5), (torch.bfloat16, 2.5e-2)],
)
@pytest.mark.parametrize("invalids", [False, True])
def test_linear_cross_entropy_kl_backward(
    dtype: torch.dtype,
    error_tol: float,
    invalids: bool,
):
    torch.cuda.manual_seed(0)

    batch_shape = (2, 19)
    vocab_size = 257
    hidden_dim = 64
    alpha = 0.6

    student_h = torch.randn((*batch_shape, hidden_dim), device="cuda", dtype=dtype) / (hidden_dim**0.5)
    teacher_h = torch.randn((*batch_shape, hidden_dim), device="cuda", dtype=dtype) / (hidden_dim**0.5)
    student_c = torch.randn((vocab_size, hidden_dim), device="cuda", dtype=dtype)
    teacher_c = torch.randn((vocab_size, hidden_dim), device="cuda", dtype=dtype)
    targets = torch.randint(0, vocab_size, size=batch_shape, device="cuda")

    if invalids:
        invalid_mask = torch.rand(batch_shape, device="cuda") < 0.25
        targets = torch.where(invalid_mask, torch.full_like(targets, IGNORE_INDEX), targets)

    ref_grad_h, ref_grad_c = _reference_grads(
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        alpha=alpha,
    )

    student_h = student_h.detach().clone().requires_grad_(True)
    student_c = student_c.detach().clone().requires_grad_(True)
    teacher_h = teacher_h.detach().clone().requires_grad_(True)
    teacher_c = teacher_c.detach().clone().requires_grad_(True)

    loss = linear_cross_entropy_kl(
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        alpha=alpha,
        reduction="mean",
    )
    loss.backward()

    assert student_h.grad is not None
    assert student_c.grad is not None
    assert teacher_h.grad is None
    assert teacher_c.grad is None
    assert torch.allclose(student_h.grad.float(), ref_grad_h.float(), atol=error_tol, rtol=error_tol)
    assert torch.allclose(student_c.grad.float(), ref_grad_c.float(), atol=error_tol, rtol=error_tol)


@skip_no_cuda
def test_linear_cross_entropy_kl_backward_small_batch():
    torch.cuda.manual_seed(0)

    batch_shape = (1, 4)
    vocab_size = 257
    hidden_dim = 64
    alpha = 0.6

    student_h = torch.randn((*batch_shape, hidden_dim), device="cuda", dtype=torch.bfloat16) / (hidden_dim**0.5)
    teacher_h = torch.randn((*batch_shape, hidden_dim), device="cuda", dtype=torch.bfloat16) / (hidden_dim**0.5)
    student_c = torch.randn((vocab_size, hidden_dim), device="cuda", dtype=torch.bfloat16)
    teacher_c = torch.randn((vocab_size, hidden_dim), device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, vocab_size, size=batch_shape, device="cuda")

    ref_grad_h, ref_grad_c = _reference_grads(
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        alpha=alpha,
    )

    student_h = student_h.detach().clone().requires_grad_(True)
    student_c = student_c.detach().clone().requires_grad_(True)
    teacher_h = teacher_h.detach().clone().requires_grad_(True)
    teacher_c = teacher_c.detach().clone().requires_grad_(True)

    loss = linear_cross_entropy_kl(
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        alpha=alpha,
        reduction="mean",
    )
    loss.backward()

    assert student_h.grad is not None
    assert student_c.grad is not None
    assert teacher_h.grad is None
    assert teacher_c.grad is None
    assert torch.allclose(student_h.grad.float(), ref_grad_h.float(), atol=2.5e-2, rtol=2.5e-2)
    assert torch.allclose(student_c.grad.float(), ref_grad_c.float(), atol=2.5e-2, rtol=2.5e-2)


@skip_no_cuda
def test_linear_cross_entropy_kl_all_ignored():
    torch.cuda.manual_seed(0)

    student_h = torch.randn((2, 8, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    teacher_h = torch.randn((2, 8, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    student_c = torch.randn((64, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    teacher_c = torch.randn((64, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.full((2, 8), IGNORE_INDEX, device="cuda", dtype=torch.int64)

    loss, ce_loss, kl_loss = linear_cross_entropy_kl(
        student_h,
        student_c,
        targets,
        teacher_h,
        teacher_c,
        reduction="mean",
        return_components=True,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert ce_loss.item() == 0.0
    assert kl_loss.item() == 0.0
    assert student_h.grad is not None
    assert student_c.grad is not None
    assert teacher_h.grad is None
    assert teacher_c.grad is None
    assert torch.count_nonzero(student_h.grad) == 0
    assert torch.count_nonzero(student_c.grad) == 0


@skip_no_cuda
def test_linear_cross_entropy_kl_rejects_mismatched_hidden_sizes():
    student_h = torch.randn((2, 4, 32), device="cuda", dtype=torch.float32)
    teacher_h = torch.randn((2, 4, 48), device="cuda", dtype=torch.float32)
    student_c = torch.randn((64, 32), device="cuda", dtype=torch.float32)
    teacher_c = torch.randn((64, 48), device="cuda", dtype=torch.float32)
    targets = torch.randint(0, 64, size=(2, 4), device="cuda")

    with pytest.raises(ValueError, match="must have identical shape"):
        linear_cross_entropy_kl(student_h, student_c, targets, teacher_h, teacher_c)