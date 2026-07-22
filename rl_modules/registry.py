from rl_modules.rl_module import RLModule
from rl_modules.marv_rl.marv_rl_module import MarvRLModule
from rl_modules.pan_d3qn.pan_d3qn_module import PanD3QNModule

# Maps CrossingEnvCfg.module_name -> RLModule subclass. Add new reward implementations here.
RLMODULE_REGISTRY: dict[str, type[RLModule]] = {
    "marv_rl": MarvRLModule,
    "pan_d3qn": PanD3QNModule,
}
