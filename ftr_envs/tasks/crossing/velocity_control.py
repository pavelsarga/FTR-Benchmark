"""Forward-velocity and heading controllers for flipper-only policies.

A deployed flipper policy never drives: in Gazebo (and on the robot) auto_ride or an operator
owns /cmd_vel and the policy only moves the flippers. Training it with [v, w, flippers] actions
evaluates it in a loop it never trained in. With `velocity_controller` set, FtrEnv exposes only
the flipper actions to the policy and fills v and w in itself, every control step:

    v  from a VelocityController   const | random | il (the pretrained velocity network)
    w  from the auto_ride heading law, ported: w = clamp(gain * bearing error, +-max_w), with the
       bearing measured to the env's goal (auto_ride aims at the next obstacle's beginning; on a
       straight FTR tile that is the same line)

The applied [v, w] is written into the env's 6-D action/last_action buffers, so every reward
term and the "previous action" observation see what was actually executed.

ILVelocity consumes the same features the Gazebo dataset is built from
(misc/velocity_il/extract_velocity_il.py), as a history of `history` control steps:
    hm      [N, T, 45, 21]  robot-centric heightmap relative to (base z - wheel radius), the
                            elevation-map-emulated one when heightmap_emulation is on
    proprio [N, T, 10]      v measured, v commanded at the previous step, flipper velocity x 4
                            (measured: position change over the last step / step_dt, rad/s),
                            flipper angle x 4 (logical convention: positive = up, rad)
and the network file is a TorchScript module mapping (hm, proprio) -> v [N] with its input
normalisation baked in (written by flipper_training's train_velocity_il.py).
"""
import math

import torch

PROPRIO_DIM = 10


class VelocityController:
    """Base: per-env state lives in the subclass; reset(env_ids) at episode start."""

    def __init__(self, num_envs: int, device):
        self.num_envs = num_envs
        self.device = device

    def reset(self, env_ids):
        pass

    def __call__(self, hm_hist: torch.Tensor, proprio_hist: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ConstVelocity(VelocityController):
    def __init__(self, num_envs, device, value: float = 0.3):
        super().__init__(num_envs, device)
        self.value = float(value)

    def __call__(self, hm_hist, proprio_hist):
        return torch.full((self.num_envs,), self.value, device=self.device)


class RandomVelocity(VelocityController):
    """Speed drawn uniformly from [low, high] per episode, redrawn mid-episode with probability
    change_prob per step -- the stand-in before the IL network exists, so the flipper policy
    already meets varying speeds it does not choose."""

    def __init__(self, num_envs, device, low: float = 0.2, high: float = 0.6, change_prob: float = 0.0):
        super().__init__(num_envs, device)
        self.low, self.high, self.change_prob = float(low), float(high), float(change_prob)
        self.v = torch.empty(num_envs, device=device).uniform_(self.low, self.high)

    def reset(self, env_ids):
        self.v[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(self.low, self.high)

    def __call__(self, hm_hist, proprio_hist):
        if self.change_prob > 0:
            redraw = torch.rand(self.num_envs, device=self.device) < self.change_prob
            self.v[redraw] = torch.empty(int(redraw.sum()), device=self.device).uniform_(self.low, self.high)
        return self.v.clone()


class ILVelocity(VelocityController):
    def __init__(self, num_envs, device, weights: str, v_max: float = 0.7):
        super().__init__(num_envs, device)
        self.net = torch.jit.load(weights, map_location=device).eval()
        self.v_max = float(v_max)

    @torch.no_grad()
    def __call__(self, hm_hist, proprio_hist):
        return self.net(hm_hist, proprio_hist).reshape(-1).clamp(0.0, self.v_max)


class AgentVelocity(ILVelocity):
    """type rl: the speed comes from a learning agent the trainer attaches (flipper_training's
    velocity_rl.VelocityAgent, co-trained with the flipper policy). Until one is attached -- eval
    and validation subprocesses -- it is the exported net `weights`, exactly as type il.

    The env also calls end_step() once per control step, after its dones and before its resets,
    so the agent can close the step's transition with its own reward."""

    def __init__(self, num_envs, device, weights: str, v_max: float = 1.0):
        super().__init__(num_envs, device, weights, v_max)
        self.agent = None

    def attach(self, agent):
        self.agent = agent

    def __call__(self, hm_hist, proprio_hist):
        if self.agent is None:
            return super().__call__(hm_hist, proprio_hist)
        return self.agent.act(hm_hist, proprio_hist)

    def reset(self, env_ids):
        if self.agent is not None:
            self.agent.reset(env_ids)

    def end_step(self, positions_xy, ang_vel_body, success, fail, timeout, abort=None):
        if self.agent is not None:
            self.agent.end_step(positions_xy, ang_vel_body, success, fail, timeout, abort)


def make_velocity_controller(spec: dict, num_envs: int, device) -> VelocityController:
    spec = dict(spec)
    kind = spec.pop("type")
    spec.pop("history", None)
    classes = {"const": ConstVelocity, "random": RandomVelocity, "il": ILVelocity, "rl": AgentVelocity}
    if kind not in classes:
        raise ValueError(f"velocity_controller.type {kind!r}: expected one of {sorted(classes)}")
    return classes[kind](num_envs, device, **spec)


class FeatureHistory:
    """Rolling [N, T, ...] buffers of the velocity controller's inputs. A reset env's history is
    filled with its first frame, like the deployment node does before T frames exist."""

    def __init__(self, num_envs: int, history: int, hm_shape, device):
        self.T = history
        self.hm = torch.zeros(num_envs, history, *hm_shape, device=device)
        self.proprio = torch.zeros(num_envs, history, PROPRIO_DIM, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids):
        self.fresh[env_ids] = True

    def push(self, hm: torch.Tensor, proprio: torch.Tensor):
        self.hm = torch.roll(self.hm, -1, dims=1)
        self.proprio = torch.roll(self.proprio, -1, dims=1)
        self.hm[:, -1] = hm
        self.proprio[:, -1] = proprio
        if self.fresh.any():
            f = self.fresh
            self.hm[f] = hm[f].unsqueeze(1).expand(-1, self.T, *hm.shape[1:])
            self.proprio[f] = proprio[f].unsqueeze(1).expand(-1, self.T, proprio.shape[-1])
            self.fresh[:] = False


def bearing_heading(positions: torch.Tensor, yaw: torch.Tensor, targets: torch.Tensor,
                    gain: float = 1.5, max_w: float = 1.0) -> torch.Tensor:
    """auto_ride's heading law: w = clamp(gain * wrap(bearing to target - yaw), +-max_w) [rad/s]."""
    bearing = torch.atan2(targets[:, 1] - positions[:, 1], targets[:, 0] - positions[:, 0])
    err = torch.remainder(bearing - yaw + math.pi, 2 * math.pi) - math.pi
    return (gain * err).clamp(-max_w, max_w)
