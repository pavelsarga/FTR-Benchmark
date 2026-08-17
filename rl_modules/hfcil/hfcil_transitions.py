"""The demonstrator's state machine, transcribed verbatim from the MARV reactive flipper
controller's mode 8 ("StateMachine") in
`marv_flipper_controller/src/Control.cpp` (case 8:).

This file is the SINGLE SOURCE OF TRUTH for the state graph. Every other file in this
module derives its legal-action masks from LEGAL_SUCCESSORS here — nothing re-declares the
graph, so a change in the C++ controller needs editing exactly one table.

Why the graph is enforced structurally rather than learned
----------------------------------------------------------
In Control.cpp the operator cannot jump to an arbitrary state: each state exposes an
ordered list of at most three successors, and the three mode buttons select `next[0]`,
`next[1]`, `next[2]` respectively. A button whose index exceeds the current state's list
does nothing. So an illegal transition is not "rare in the demonstrations" — it is
*impossible to demonstrate*. Training a classifier over all 8 states with a plain softmax
would let it place probability mass on transitions the demonstrator could never perform,
and at rollout time it would eventually take one. Instead we mask illegal logits to -inf
in both training and inference (see hfcil_policy.py), which makes the constraint a
property of the architecture rather than something the network has to infer from data.

Self-transitions
----------------
Control.cpp holds the current state whenever no button is pressed, which is the
overwhelming majority of ticks. "Stay" is therefore always legal and is represented by the
state's own index being present in its mask — it is NOT one of the `next` entries in the
C++ source (those are strictly the button-selectable successors).

BALLERINA (7)
-------------
No state lists 7 as a successor, so it is unreachable through the button transitions —
it is entered out-of-band (the operator switches into it by other means) and then exits to
{N, AR, DR}. Its outgoing edges are kept faithful here, but a demonstration dataset
collected purely by driving will contain no *entries* into it. See STATES_UNREACHABLE.
"""

# Index -> (short name, full name). Order is Control.cpp's state_poses[] order and is
# load-bearing: the integer labels in a demonstration dataset are these indices.
STATE_SHORT_NAMES = ("N", "AF", "AS", "AR", "DF", "DS", "DR", "B")
STATE_FULL_NAMES = (
    "NEUTRAL",
    "ASCENDING_FRONT",
    "ASCENDING_STAIRS",
    "ASCENDING_REAR",
    "DESCENDING_FRONT",
    "DESCENDING_STAIRS",
    "DESCENDING_REAR",
    "BALLERINA",
)
NUM_STATES = len(STATE_SHORT_NAMES)

# Button-selectable successors, exactly as the `switch (statemachine_state)` block reads.
# The ORDER matters: position i is the state reached by button i (first/second/third).
BUTTON_SUCCESSORS: dict[int, tuple[int, ...]] = {
    0: (1, 4),      # N  -> AF, DF
    1: (0, 2, 3),   # AF -> N, AS, AR
    2: (3,),        # AS -> AR
    3: (0,),        # AR -> N
    4: (0, 5, 6),   # DF -> N, DS, DR
    5: (6,),        # DS -> DR
    6: (0,),        # DR -> N
    7: (0, 3, 6),   # B  -> N, AR, DR
}

# What the policy may output given the current state: stay, or any button successor.
LEGAL_SUCCESSORS: dict[int, tuple[int, ...]] = {
    s: tuple(sorted({s, *succ})) for s, succ in BUTTON_SUCCESSORS.items()
}

# States no other state can transition into (see module docstring).
STATES_UNREACHABLE = tuple(
    s for s in range(NUM_STATES)
    if all(s not in succ for src, succ in BUTTON_SUCCESSORS.items() if src != s)
)

# Flipper pose per state, raw radians, verbatim from Control.cpp's state_poses[], in that
# file's field order (front_left, front_right, rear_left, rear_right). Not used to drive
# the sim directly — FtrEnv takes normalised commands and HFC's own decoder owns that
# mapping — but kept here so a collected demonstration can be checked against the pose the
# controller actually commanded for its labelled state.
STATE_POSES_RAD: dict[int, tuple[float, float, float, float]] = {
    0: (-2.0, -2.0, 1.5, 1.5),
    1: (-0.4, -0.4, 0.0, 0.0),
    2: (0.0, 0.0, 0.05, 0.05),
    3: (0.1, 0.1, -0.6, -0.5),
    4: (0.35, 0.35, -0.7, -0.7),
    5: (0.0, 0.0, 0.05, 0.05),
    6: (-0.3, -0.3, 0.4, 0.4),
    7: (1.58, 1.58, -1.58, -1.58),
}


def legal_mask(num_states: int = NUM_STATES):
    """(num_states, num_states) bool tensor, mask[s, s'] = True iff s -> s' is legal.

    Built lazily so this module stays importable without torch (the dataset tooling that
    validates a recording runs outside the Isaac Sim environment).
    """
    import torch

    mask = torch.zeros(num_states, num_states, dtype=torch.bool)
    for s, succ in LEGAL_SUCCESSORS.items():
        for s_next in succ:
            mask[s, s_next] = True
    return mask


def is_legal(state: int, next_state: int) -> bool:
    return next_state in LEGAL_SUCCESSORS[state]
