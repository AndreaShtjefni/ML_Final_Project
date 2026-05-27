"""Combine four v8 data files and filter dead frames.

A dead frame is one where the human was idle:
  abs(throttle) < 0.05 AND abs(steering) < 0.05

Keeping dead frames teaches the bot to output [0, 0] and freeze (v3 lesson).
This script removes them automatically.
"""
import numpy as np

d1 = np.load('data_v8a.npz')
d2 = np.load('data_v8b.npz')
d3 = np.load('data_v8c.npz')
d4 = np.load('data_v8d.npz')

print(f"v8a samples: {d1['states'].shape[0]}")
print(f"v8b samples: {d2['states'].shape[0]}")
print(f"v8c samples: {d3['states'].shape[0]}")
print(f"v8d samples: {d4['states'].shape[0]}")

combined_states   = np.vstack([d1['states'],    d2['states'],    d3['states'],    d4['states']])
combined_actions  = np.vstack([d1['actions'],   d2['actions'],   d3['actions'],   d4['actions']])
combined_positions = np.vstack([d1['positions'], d2['positions'], d3['positions'], d4['positions']])

print(f"Combined before filter: {combined_states.shape[0]} samples")

# Remove dead frames (both throttle and steering near zero)
dead = (np.abs(combined_actions[:, 0]) < 0.05) & (np.abs(combined_actions[:, 1]) < 0.05)
print(f"Dead frames removed:    {dead.sum()}")

combined_states   = combined_states[~dead]
combined_actions  = combined_actions[~dead]
# positions is polled at 5 Hz vs 20 Hz for states — different length, not filtered

print(f"Combined after filter:  {combined_states.shape[0]} samples")

np.savez('data_v8-combined.npz',
         states=combined_states,
         actions=combined_actions,
         positions=combined_positions,
         seed=d1['seed'])

print("Saved data_v8-combined.npz")
