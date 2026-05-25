"""Combine two data files into one larger dataset."""
import numpy as np

d1 = np.load('data_v5a.npz')
d2 = np.load('data_v5b.npz')

combined_states   = np.vstack([d1['states'],    d2['states']])
combined_actions  = np.vstack([d1['actions'],   d2['actions']])
combined_positions = np.vstack([d1['positions'], d2['positions']])

np.savez('data_v5-combined.npz',
         states=combined_states,
         actions=combined_actions,
         positions=combined_positions,
         seed=d1['seed'])

print(f"v5a samples:      {d1['states'].shape[0]}")
print(f"v5b samples:      {d2['states'].shape[0]}")
print(f"combined samples: {combined_states.shape[0]}")
print(f"Saved data_v5-combined.npz")
