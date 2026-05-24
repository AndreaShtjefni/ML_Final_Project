# Drive2Win — Project Notes

## Current Status
- On iteration v1
- `data_v1.npz` is GOOD — re-collected, bot did not get stuck

## Rules (never break these)
- Never edit `03_benchmark.py` — it defines fair comparison, editing it makes numbers meaningless
- All training data must come from the user's own driving, no sharing with classmates
- No external pretrained models

## Grading Reminder
- 50% process grade: git log + benchmarks/ folder, need 6–10 iterations with real commit messages
- 50% tournament: checkpoints passed across 5 rounds (3 without obstacles, 2 with)
- Commit message format: tag name + what changed + prediction + actual result

## What Good Data Looks Like
- `fig_actions_<tag>.png`: steering histogram should be symmetric (left AND right turns)
- `fig_heading_<tag>.png`: must slope downward — if flat, network can't learn to navigate
- Dataset must include wall recovery: drive into walls on purpose and drive away

## Benchmark Outputs (in benchmarks/)
- `<tag>.json` — completion rate, median lap, crashes, max checkpoints
- `<tag>_overlay.png` — user driving (gray) vs bot (blue) — most important plot, shows where data is thin
- `<tag>_paths.png` — top-down map of all 5 runs
- `<tag>_progress.png` — checkpoints per run (high variance = inconsistent model)
