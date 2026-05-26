"""Combine three v5 data files and filter dead frames.

A dead frame is one where the human was idle:
  abs(throttle) < 0.05 AND abs(steering) < 0.05

Keeping dead frames teaches the bot to output [0, 0] and freeze (v3 lesson).
This script removes them automatically.
"""
import numpy as np

d1 = np.load('data_v5a.npz')
d2 = np.load('data_v5b.npz')
d3 = np.load('data_v5c.npz')

print(f"v5a samples: {d1['states'].shape[0]}")
print(f"v5b samples: {d2['states'].shape[0]}")
print(f"v5c samples: {d3['states'].shape[0]}")

combined_states   = np.vstack([d1['states'],    d2['states'],    d3['states']])
combined_actions  = np.vstack([d1['actions'],   d2['actions'],   d3['actions']])
combined_positions = np.vstack([d1['positions'], d2['positions'], d3['positions']])

print(f"Combined before filter: {combined_states.shape[0]} samples")

# Remove dead frames (both throttle and steering near zero)
dead = (np.abs(combined_actions[:, 0]) < 0.05) & (np.abs(combined_actions[:, 1]) < 0.05)
print(f"Dead frames removed:    {dead.sum()}")

combined_states   = combined_states[~dead]
combined_actions  = combined_actions[~dead]
# positions is polled at 5 Hz vs 20 Hz for states — different length, not filtered

print(f"Combined after filter:  {combined_states.shape[0]} samples")

np.savez('data_v5-combined.npz',
         states=combined_states,
         actions=combined_actions,
         positions=combined_positions,
         seed=d1['seed'])

print("Saved data_v5-combined.npz")
