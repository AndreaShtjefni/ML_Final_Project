"""Tournament bot — plugs nav_v6-long.npz into RoomBot.

Usage:
    python my_bot.py --room <room-name> --name <your-name>

On tournament day the instructor will announce the room name.
"""
from __future__ import annotations
import argparse
import sys
import numpy as np

# Add parent path so game_client can be imported if needed
sys.path.insert(0, ".")

from game_client import RoomBot
from drive2win import nn as nn_mod
from drive2win.normalize import normalize_states

WEIGHTS = "nav_v6-long.npz"
SERVER  = "https://ml.ferit.tech"

# Load weights once at startup
_w = nn_mod.load(WEIGHTS)


def controller(obs):
    """Convert obs dict -> (throttle, steering) using our trained MLP."""
    nav = obs["navigation"]

    # Build the 11-feature vector (same layout as training, checkpoint_distance dropped)
    raw = np.array([
        obs["speed"],
        nav["heading_error"],
        # skip nav["distance"] — checkpoint_distance was removed in v4
        *obs["rays"],           # 8 raycasts
        obs["ground_friction"],
    ], dtype=np.float32)        # shape (11,)

    x = normalize_states(raw[None, :])[0]   # normalize to [-1, 1]
    out = nn_mod.forward(x, _w)             # shape (1, 2) or (2,)
    out = np.asarray(out).reshape(-1)
    throttle = float(np.clip(out[0], -1.0, 1.0))
    steering = float(np.clip(out[1], -1.0, 1.0))
    return throttle, steering


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True, help="Tournament room name")
    ap.add_argument("--name", default="AndreaShtjefni", help="Bot display name")
    ap.add_argument("--host", default=SERVER)
    args = ap.parse_args()

    print(f"Connecting to room '{args.room}' as '{args.name}'...")
    print(f"Weights: {WEIGHTS}")

    bot = RoomBot(args.host, room=args.room, name=args.name)
    standings = bot.run(controller, hz=20.0)
    print("\n=== Final Standings ===")
    for s in standings:
        print(s)


if __name__ == "__main__":
    main()
