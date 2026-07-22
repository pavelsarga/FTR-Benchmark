import torch
import torch.nn as nn

from marv_rl_training.observations import ObservationEncoder
from marv_rl_training.policies import MLP


class MarvRLHeightmapEncoder(ObservationEncoder):
    def __init__(
        self,
        img_shape: tuple[int, int],
        output_dim: int,
        activate_output: bool = False,
        **kwargs,
    ):
        super(MarvRLHeightmapEncoder, self).__init__(output_dim)
        self.img_shape = img_shape  # Keep for reference if needed, but not used in layer defs anymore
        # Define the sequential convolutional layers
        self.encoder = nn.Sequential(
            # Layer 1: Similar to the original stem but using 3x3 kernel
            # Input: (B, 1, 45, 21)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0, dilation=2),
            nn.ReLU(inplace=True),
            # Output: (B, 16, 21, 9)
            # Layer 2: Downsample, increase channels
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            # Output: (B, 32, 10, 4)
            # Layer 3: Downsample, increase channels
            nn.Conv2d(32, 64, kernel_size=(4,2), stride=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            # Output: (B, 64, 7, 3)
            nn.AdaptiveAvgPool2d((2, 2)),  # Pool to 2x2 spatial dimensions
            nn.Flatten(),  # Flatten features -> (B, 64 * 2 * 2)
            nn.Linear(4 * 64, output_dim),  # Linear layer -> (B, output_dim)
            nn.ReLU(inplace=True) if activate_output else nn.Identity(),
        )

    def forward(self, hm):
        # Handle potential time dimension (same as before)
        if hm.ndim > 4:
            B, T = hm.shape[:2]
            # Input shape expected: (B, T, C, H, W)
            C, H, W = hm.shape[2:]
            hm = hm.view(B * T, C, H, W)  # Use view for efficiency
            y_ter = self.encoder(hm)
            # Output shape expected: (B, T, output_dim)
            y_ter = y_ter.view(B, T, -1)
        else:
            # Input shape expected: (B, C, H, W)
            y_ter = self.encoder(hm)
            # Output shape expected: (B, output_dim)
        return y_ter


class MarvRLCNNFlatEncoder(ObservationEncoder):
    HM_SIZE = (45, 21)   # 945 values
    HM_DIM = 945

    def __init__(self, output_dim: int, state_dim: int = 21, cnn_output_dim: int = 128, state_proj_dim: int | None = None, input_dim: int | None = None, **mlp_kwargs):
        super().__init__(output_dim)
        self.state_dim = state_dim
        self.cnn = MarvRLHeightmapEncoder(self.HM_SIZE, output_dim=cnn_output_dim)
        # Optional learned projection to bring state up to a comparable scale before fusion.
        # Without this, the 21-dim state is structurally dominated by the 128-dim CNN output.
        if state_proj_dim is not None:
            self.state_proj = MLP(in_dim=state_dim, out_dim=state_proj_dim, hidden_dim=state_proj_dim, num_hidden=1, layernorm=False, activate_last_layer=True)
            fusion_dim = cnn_output_dim + state_proj_dim
        else:
            self.state_proj = None
            fusion_dim = cnn_output_dim + state_dim
        self.mlp = MLP(in_dim=fusion_dim, out_dim=output_dim, activate_last_layer=True, **mlp_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, HM_DIM + state_dim]
        hm = x[..., :self.HM_DIM].view(*x.shape[:-1], 1, 45, 21)  # [N,1,45,21]
        state = x[..., self.HM_DIM:]                                # [N, state_dim]
        latent = self.cnn(hm)                                        # [N,cnn_output_dim]
        if self.state_proj is not None:
            state = self.state_proj(state)                           # [N, state_proj_dim]
        return self.mlp(torch.cat([latent, state], dim=-1))
