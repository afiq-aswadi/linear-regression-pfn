import torch
import pytest

from baselines import dmmse_predictor, dmmse_predictor_vectorized
from samplers.tasks import DiscreteTaskDistribution


@pytest.mark.parametrize("B,K,D,M", [(2, 5, 3, 4), (1, 3, 2, 5)])
def test_vectorized_matches_reference(B, K, D, M):
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Deterministic prior
    prior = DiscreteTaskDistribution(task_size=D, num_tasks=M, device=str(device))
    tasks = torch.randn(M, D, device=device)
    prior.tasks = tasks

    xs = torch.randn(B, K, D, device=device)
    ys = torch.randn(B, K, 1, device=device)

    noise_var = 0.25

    ys_ref, ws_ref = dmmse_predictor(xs, ys, prior, noise_var, return_ws_hat=True)
    ys_vec, ws_vec = dmmse_predictor_vectorized(
        xs, ys, prior, noise_variance=noise_var, return_ws_hat=True
    )

    assert ys_ref.shape == ys_vec.shape == (B, K, 1)
    assert ws_ref.shape == ws_vec.shape == (B, K, D)
    assert torch.allclose(ys_ref, ys_vec, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ws_ref, ws_vec, rtol=1e-5, atol=1e-6)


def test_vectorized_bounds_match_reference():
    torch.manual_seed(1)
    B, K, D, M = 3, 6, 4, 7
    device = torch.device("cpu")

    prior = DiscreteTaskDistribution(task_size=D, num_tasks=M, device=str(device))
    prior.tasks = torch.randn(M, D, device=device)

    xs = torch.randn(B, K, D, device=device)
    ys = torch.randn(B, K, 1, device=device)

    noise_var = 0.5
    y_min, y_max = -0.3, 0.4

    ys_ref = dmmse_predictor(
        xs, ys, prior, noise_var,
        return_ws_hat=False,
        bound_by_y_min_max=True,
        y_min=y_min,
        y_max=y_max,
    )
    ys_vec = dmmse_predictor_vectorized(
        xs, ys, prior, noise_variance=noise_var,
        return_ws_hat=False,
        bound_by_y_min_max=True,
        y_min=y_min,
        y_max=y_max,
    )

    assert torch.allclose(ys_ref, ys_vec, rtol=1e-5, atol=1e-6)
    assert torch.all(ys_vec >= y_min - 1e-6)
    assert torch.all(ys_vec <= y_max + 1e-6)


def test_zero_context_uniform_prior_mean():
    torch.manual_seed(2)
    B, K, D, M = 2, 4, 3, 5
    device = torch.device("cpu")

    prior = DiscreteTaskDistribution(task_size=D, num_tasks=M, device=str(device))
    prior.tasks = torch.randn(M, D, device=device)

    xs = torch.randn(B, K, D, device=device)
    ys = torch.randn(B, K, 1, device=device)

    _, ws_vec = dmmse_predictor_vectorized(xs, ys, prior, return_ws_hat=True)
    mean_task = prior.tasks.mean(dim=0)

    # k=0 should use exclusive prefix (no data) -> uniform -> mean task
    assert torch.allclose(ws_vec[:, 0, :], mean_task.expand(B, -1), rtol=1e-5, atol=1e-6)


