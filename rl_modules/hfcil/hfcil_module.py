"""HFC-IL: observations + reward for the imitation-learned HFC controller.

Deliberately subclasses HFCModule rather than copying it. Every other module in this
project owns its observation/reward independently (no cross-module imports) — this one is
the exception, and the exception is the point: the classifier is trained offline on
demonstrations recorded in HFCObservation's exact 24-D layout, so if hfcil's observation
ever drifted from hfc's by a single field, every recorded demonstration would silently
become mislabelled input. Inheriting makes that drift impossible instead of merely
discouraged.

Reward is inherited too, and is only used for *evaluation* — imitation learning never reads
it (train_hfcil.py optimises cross-entropy against the demonstrator's next state). It is
what lets an hfcil checkpoint be scored by the same eval path, and against the same
numbers, as hfc/marv_rl.
"""
import logging

from rl_modules.hfc.hfc_module import HFCModule

_log = logging.getLogger(__name__)


class HFCILModule(HFCModule):
    """Identical observations and reward to HFCModule — see this file's docstring.

    The two modules differ only in how their policy's transition classifier is trained
    (supervised on demonstrations here, PPO in hfc), which lives entirely in
    hfcil_policy.py / train_hfcil.py, not here.
    """

    pass
