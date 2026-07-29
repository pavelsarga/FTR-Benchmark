import torch
import torch.nn as nn


class CREPSLowerLevelPolicy(nn.Module):
    """Deterministic linear lower-level policy from Pecka et al. 2016, "Autonomous
    Flipper Control with Safety Constraints":
        front = omega1 + omega2*pitch + omega3*height_ahead
        rear  = omega4 + omega5*pitch + omega6*height_ahead

    `omega` is a per-env BUFFER of shape (num_envs, 6), NOT an nn.Parameter — there is no
    gradient anywhere in CREPS. It is overwritten wholesale via set_omega() at the start
    of every outer iteration by train_creps.py, sampled from the current upper-level
    distribution q(omega) = N(mu, Sigma) (creps_algorithm.py). Within an iteration/episode
    it stays fixed per env, exactly matching the paper's "lower-level policy samples,
    evaluated by executing them" design.

    forward() returns the raw 4-D action [0, 0, front, rear] UNCLAMPED — front/rear are
    NOT clipped to [-1,1] here. FtrEnv stores this exact value in env.last_action before
    internally clamping it (ftr_env.py's position-control mapping) to compute the actual
    flipper target — so leaving it unclamped here is required for env.last_action to
    genuinely reflect "what the linear policy wanted" (the paper's mechanical-constraint
    indicator C_sw is computed by train_creps.py from this raw value; clamping here would
    make every rollout appear mechanically compliant regardless of omega).

    v and w (track linear/angular velocity) are always 0: the env's fixed_forward_vel
    override replaces v with a constant regardless of what's commanded here (the paper's
    "robot automatically driven forward by a constant speed"), and the paper's task has
    no steering (w=0).
    """

    def __init__(self, num_envs: int, device):
        super().__init__()
        self.register_buffer("omega", torch.zeros(num_envs, 6, device=device))

    def set_omega(self, omega: torch.Tensor) -> None:
        assert omega.shape == self.omega.shape, f"expected {tuple(self.omega.shape)}, got {tuple(omega.shape)}"
        self.omega.copy_(omega.to(device=self.omega.device, dtype=self.omega.dtype))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pitch, height = obs[..., 0], obs[..., 1]
        front = self.omega[:, 0] + self.omega[:, 1] * pitch + self.omega[:, 2] * height
        rear = self.omega[:, 3] + self.omega[:, 4] * pitch + self.omega[:, 5] * height
        v = torch.zeros_like(front)
        w = torch.zeros_like(front)
        return torch.stack([v, w, front, rear], dim=-1)  # (N, 4), front/rear UNCLAMPED
