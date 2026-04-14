# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import contextlib
import socket

import pytest
import torch
import torch.distributed
from torch.multiprocessing.spawn import spawn as mp_spawn

from cut_cross_entropy import VocabParallelOptions, linear_cross_entropy_kl
from cut_cross_entropy.constants import IGNORE_INDEX


def find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", 0))
        _, port = sock.getsockname()
        return port


def _target_fn_test_linear_cross_entropy_kl_vp(
    rank: int,
    world_size: int,
    port: int,
    dtype: torch.dtype,
    error_tol: float,
    invalids: bool,
):
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    store = torch.distributed.TCPStore(
        "localhost", port, world_size=world_size, is_master=rank == 0
    )
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        store=store,
        world_size=world_size,
        rank=rank,
    )
    store = None

    batch_shape = (3, 17)
    vocab_size = 509
    hidden_dim = 64
    alpha = 0.37

    student_h = torch.randn((*batch_shape, hidden_dim), device=device, dtype=dtype)
    teacher_h = torch.randn((*batch_shape, hidden_dim), device=device, dtype=dtype)
    student_c = torch.randn((vocab_size, hidden_dim), device=device, dtype=dtype)
    teacher_c = torch.randn((vocab_size, hidden_dim), device=device, dtype=dtype)
    targets = torch.randint(0, vocab_size, size=batch_shape, device=device)

    if invalids:
        invalid_mask = torch.rand(batch_shape, device=device) < 0.2
        targets = torch.where(invalid_mask, torch.full_like(targets, IGNORE_INDEX), targets)

    torch.distributed.broadcast(student_h, src=0)
    torch.distributed.broadcast(teacher_h, src=0)
    torch.distributed.broadcast(student_c, src=0)
    torch.distributed.broadcast(teacher_c, src=0)
    torch.distributed.broadcast(targets, src=0)

    ref_student_h = student_h.detach().clone().requires_grad_(True)
    ref_student_c = student_c.detach().clone().requires_grad_(True)
    ref_teacher_h = teacher_h.detach().clone().requires_grad_(True)
    ref_teacher_c = teacher_c.detach().clone().requires_grad_(True)

    ref_all, ref_ce, ref_kl = linear_cross_entropy_kl(
        ref_student_h,
        ref_student_c,
        targets,
        ref_teacher_h,
        ref_teacher_c,
        alpha=alpha,
        reduction="none",
        return_components=True,
    )

    ref_loss = linear_cross_entropy_kl(
        ref_student_h,
        ref_student_c,
        targets,
        ref_teacher_h,
        ref_teacher_c,
        alpha=alpha,
        reduction="mean",
    )
    ref_loss.backward()

    assert ref_student_h.grad is not None
    assert ref_student_c.grad is not None
    assert ref_teacher_h.grad is None
    assert ref_teacher_c.grad is None

    vocab_parallel_options = VocabParallelOptions.from_vocab(vocab_size)
    vp_start = vocab_parallel_options.start
    vp_stop = vocab_parallel_options.stop

    vp_student_h = student_h.detach().clone().requires_grad_(True)
    vp_teacher_h = teacher_h.detach().clone().requires_grad_(True)
    vp_student_c = student_c[vp_start:vp_stop].detach().clone().requires_grad_(True)
    vp_teacher_c = teacher_c[vp_start:vp_stop].detach().clone().requires_grad_(True)

    vp_all, vp_ce, vp_kl = linear_cross_entropy_kl(
        vp_student_h,
        vp_student_c,
        targets,
        vp_teacher_h,
        vp_teacher_c,
        alpha=alpha,
        reduction="none",
        return_components=True,
        vocab_parallel_options=vocab_parallel_options,
    )

    max_all_diff = (vp_all.float() - ref_all.float()).abs().max().item()
    max_ce_diff = (vp_ce.float() - ref_ce.float()).abs().max().item()
    max_kl_diff = (vp_kl.float() - ref_kl.float()).abs().max().item()

    assert torch.allclose(vp_all.float(), ref_all.float(), atol=error_tol, rtol=error_tol), (
        f"max_all_diff={max_all_diff:.6e}, max_ce_diff={max_ce_diff:.6e}, max_kl_diff={max_kl_diff:.6e}"
    )
    assert torch.allclose(vp_ce.float(), ref_ce.float(), atol=error_tol, rtol=error_tol), (
        f"max_ce_diff={max_ce_diff:.6e}"
    )
    assert torch.allclose(vp_kl.float(), ref_kl.float(), atol=error_tol, rtol=error_tol), (
        f"max_kl_diff={max_kl_diff:.6e}"
    )

    vp_loss = linear_cross_entropy_kl(
        vp_student_h,
        vp_student_c,
        targets,
        vp_teacher_h,
        vp_teacher_c,
        alpha=alpha,
        reduction="mean",
        vocab_parallel_options=vocab_parallel_options,
    )
    vp_loss.backward()

    assert vp_student_h.grad is not None
    assert vp_student_c.grad is not None
    assert vp_teacher_h.grad is None
    assert vp_teacher_c.grad is None
    max_h_diff = (vp_student_h.grad.float() - ref_student_h.grad.float()).abs().max().item()
    max_c_diff = (
        vp_student_c.grad.float() - ref_student_c.grad[vp_start:vp_stop].float()
    ).abs().max().item()

    assert torch.allclose(
        vp_student_h.grad.float(),
        ref_student_h.grad.float(),
        atol=error_tol,
        rtol=error_tol,
    ), f"max_h_grad_diff={max_h_diff:.6e}"
    assert torch.allclose(
        vp_student_c.grad.float(),
        ref_student_c.grad[vp_start:vp_stop].float(),
        atol=error_tol,
        rtol=error_tol,
    ), f"max_c_grad_diff={max_c_diff:.6e}"

    torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.parametrize("dtype,error_tol", [(torch.float32, 1e-5), (torch.bfloat16, 2.5e-2)])
@pytest.mark.parametrize("invalids", [False, True])
@pytest.mark.parametrize("nprocs", [2])
def test_linear_cross_entropy_kl_vocab_parallel(
    dtype: torch.dtype,
    error_tol: float,
    invalids: bool,
    nprocs: int,
):
    mp_spawn(
        _target_fn_test_linear_cross_entropy_kl_vp,
        args=(nprocs, find_free_port(), dtype, error_tol, invalids),
        nprocs=nprocs,
        join=True,
    )