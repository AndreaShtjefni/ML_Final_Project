"""Tournament agent with action smoothing.

Wraps the default MLP policy with Fix A — action smoothing:
  output = 0.7 * network_prediction + 0.3 * last_output

This dampens the jitter/snapping behaviour that makes the bot drive like a
madman. The network still controls the car but sudden left/right snaps are
blended out over consecutive frames.

Usage:
    python 03_benchmark.py --weights nav_v10.npz --tag v10-smooth --module drive2win.agent --seeds 42
"""
from __future__ import annotations
import numpy as np

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action


SMOOTH = 0.3   # weight given to previous output (0 = no smoothing, 1 = frozen)


def make_policy(weights_path: str):
    """Return a smoothed policy function.

    The returned function keeps internal state (last_output) between calls
    so smoothing persists across the full run.
    """
    w = nn_mod.load(weights_path)
    last_output = [np.zeros(2, dtype=np.float32)]

    def policy(state):
        x = sensors_to_input(state["sensors"])
        raw = np.array(clip_action(nn_mod.forward(x, w)), dtype=np.float32)
        smoothed = (1.0 - SMOOTH) * raw + SMOOTH * last_output[0]  # blend
        smoothed = np.clip(smoothed, -1.0, 1.0)                 # stay in bounds
        last_output[0] = smoothed
        return float(smoothed[0]), float(smoothed[1])

    return policy
