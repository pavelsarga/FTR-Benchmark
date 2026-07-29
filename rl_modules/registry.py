from rl_modules.rl_module import RLModule
from rl_modules.marv_rl.marv_rl_module import MarvRLModule
from rl_modules.atd3qn.atd3qn_module import ATD3QNModule
from rl_modules.icmd3qn.icmd3qn_module import ICMD3QNModule
from rl_modules.hfc.hfc_module import HFCModule
from rl_modules.mitriakov.mitriakov_module import MitriakovModule
from rl_modules.creps.creps_module import CREPSModule

# Maps CrossingEnvCfg.module_name -> RLModule subclass. Add new reward implementations here.
RLMODULE_REGISTRY: dict[str, type[RLModule]] = {
    "marv_rl": MarvRLModule,
    "atd3qn": ATD3QNModule,
    "icmd3qn": ICMD3QNModule,
    "hfc": HFCModule,
    "mitriakov": MitriakovModule,
    "creps": CREPSModule,
}
