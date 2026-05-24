# Drive2Win — Complete Guide to Getting the Highest Grade

---

## How You Are Graded (understand this first)

The grade splits into two equal halves:

**50% — Process Grade** (your git history + `benchmarks/` folder)
- Instructor reads your git log from start to finish
- They want to see: visible improvement, clear hypotheses, deliberate changes
- Target: **6–10 iterations**, each one a commit with a real message
- Too few iterations (3) = low grade. Trivial/empty commits = also low grade

**50% — Tournament Performance**
- Total checkpoints your bot passes across 5 rounds on the final day
- 5 rounds × 5 minutes each, terrain seed changes every round
- 3 rounds without obstacles, 2 rounds **with** obstacles
- Bonus: 1st place = +10%, 2nd = +5%, 3rd = +2% to your **overall course grade**

---

## Phase 0 — Setup (do this once)

**Install dependencies:**
```
pip install numpy matplotlib scikit-learn torch requests websocket-client
```

**Initialize a git repo** (critical — this is how you get graded):
```
git init
git add .
git commit -m "initial scaffold"
```

Run everything from the project root (the folder containing `drive2win/`).

---

## Phase 1 — Iteration 1 (Baseline, everyone does this)

### Step 1 — Collect your driving data

```
python 01_collect.py --tag v1 --seed 42
```

This runs 5 timed phases of manual driving (~6 minutes total). Drive carefully:
- Smooth laps around the course
- Tight turns
- Near obstacles
- Bad terrain
- **Recovery from walls** — this is the single most important thing to include. If your bot crashes into a wall, it needs training data on how to get unstuck. Drive into walls on purpose and drive away from them.

Output: `data_v1.npz`

**Critical:** Your training data must come from YOUR driving. No swapping with classmates.

---

### Step 2 — Implement Backpropagation (the graded coding task)

Open `02_train.py` and find `my_backward()`. This is where you fill in the chain rule. The network structure is:

```
Input (12) → z1 = x @ W1 + b1 → a1 = ReLU(z1)
           → z2 = a1 @ W2 + b2 → a2 = ReLU(z2)
           → z3 = a2 @ W3 + b3 → y = tanh(z3)
           → Loss = MSE(y, target)
```

Fill in each `...` in order:

```python
# Output layer
dy  = 2 * (y - y_target) / (n * y.shape[1])   # MSE gradient
dz3 = dy * (1 - y * y)                          # tanh derivative: 1 - tanh²
dW3 = cache["a2"].T @ dz3                        # gradient for W3
db3 = dz3.sum(axis=0)                            # gradient for b3

# Hidden layer 2
da2 = dz3 @ w["W3"].T                           # backprop through W3
dz2 = da2 * (cache["z2"] > 0)                   # ReLU derivative (mask)
dW2 = cache["a1"].T @ dz2
db2 = dz2.sum(axis=0)

# Hidden layer 1
da1 = dz2 @ w["W2"].T
dz1 = da1 * (cache["z1"] > 0)                   # ReLU derivative (mask)
dW1 = x.T @ dz1
db1 = dz1.sum(axis=0)
```

**The gradient check will fire automatically** before training starts. If any parameter shows error >= 1e-4, it prints "BUG" and stops. You must fix the bug before training is allowed. This is intentional — it verifies you actually understand backprop.

---

### Step 3 — Train

```
python 02_train.py --data data_v1.npz --tag v1
```

This runs 300 epochs, Adam optimizer, batch size 64, learning rate 0.001, 90/10 train/val split.

Outputs:
- `nav_v1.npz` — your trained weights
- `fig_loss_v1.png` — training/validation loss curves
- `fig_actions_v1.png` — histogram of your actions
- `fig_heading_v1.png` — heading error vs steering (should slope downward)

**Typical first result:** completion 1–2 out of 5 runs, median lap ~55 seconds, 1–2 crashes.

---

### Step 4 — Benchmark

```
python 03_benchmark.py --tag v1 --data data_v1.npz
```

Outputs (go into `benchmarks/`):
- `v1.json` — numbers: completion rate, median lap time, crashes, max checkpoints
- `v1_paths.png` — all 5 runs overlaid on a top-down map
- `v1_progress.png` — bar chart of checkpoints per run
- `v1_overlay.png` — your driving (gray) vs the NN's driving (blue) — the most revealing plot

**Never edit `benchmark.py`.** It defines the fair comparison. Editing it means your numbers are meaningless.

---

### Step 5 — Commit

```
git add data_v1.npz nav_v1.npz benchmarks/v1.* fig_*_v1.png
git commit -m "v1-bc: baseline behavioral cloning, 12→64→32→2 MLP, completion 2/5"
```

The commit message format matters. Include: **what you changed**, **what you predicted**, **what actually happened**.

---

## Phase 2 — Iterations 2 through N (6–10 total)

**The loop:**
1. Look at your figures — where does the bot fail?
2. Form ONE hypothesis ("I think the bot crashes because it has no recovery data")
3. Make ONE change to test that hypothesis
4. Re-collect and/or retrain
5. Benchmark with same seeds for fair comparison
6. Commit with a message explaining prediction vs. result
7. Repeat

---

## What to Change Each Iteration (in priority order)

### Highest impact — do these first

**Recovery data (iteration 2)**

The most effective single fix. Re-run `01_collect.py` but spend most of your driving time crashing into walls and recovering. Append or replace your dataset.
```
python 01_collect.py --tag v2-recovery --seed 42
```
Tag: `v2-recovery`

---

**Test on multiple seeds (do this by iteration 3 at the latest)**
```
python 03_benchmark.py --tag v2-recovery --seeds 42 7 99
```
If your bot is great on seed 42 but terrible on seed 7 and 99, you have memorized the map — not learned to drive. The tournament uses 5 different seeds. **You will fail rounds you haven't tested on.**

---

**DAgger-lite (very high impact)**

Watch your bot run live. When it fails (hits a wall, gets stuck), take over with WASD, drive the correct path for those few seconds, save those frames, and add them to your dataset. This is targeted data collection from actual failure points.

Tag example: `v4-dagger`

---

### Medium impact — good for iterations 3–6

**Deeper/wider network**

Edit `drive2win/nn.py`, change `H1, H2 = 64, 32` to something like `H1, H2 = 128, 64`. Also update `forward()`, `forward_all()`, and re-derive `backward()` in `02_train.py`.

Tag example: `v3-deepnet`

---

**Action smoothing**

Instead of predicting raw `(throttle, steering)`, predict the *change* from the last action (delta steering). Or apply a low-pass filter at inference to smooth jerky outputs. This especially helps with crashes from overcorrection.

---

**Different normalization**

Edit `drive2win/normalize.py`. For example, divide the 8 ray sensors by their per-channel standard deviation instead of a fixed `RAY_MAX`. Better normalization = more stable training.

---

**LeakyReLU instead of ReLU**

Change the activation in `drive2win/nn.py`. Only the derivative in `backward()` changes — `(cache["z2"] > 0)` becomes `np.where(cache["z2"] > 0, 1, 0.01)`. Helps with "dying ReLU" units.

---

### Advanced — for iterations 7–10 if you want top placement

**CNN on 32×32 terrain grid**

Add `drive2win/cnn.py` using PyTorch. Expose `make_policy(weights_path)`, then benchmark with `--module drive2win.cnn`. This is especially powerful for the 2 obstacle rounds.

Tag: `v5-cnn-hybrid`

**Hybrid CNN + MLP**

Concatenate CNN features with the 12-vector input before the final fully-connected layer. Almost always beats either alone on obstacle rounds.

**Ensemble**

Train the same architecture twice with different random seeds, average their actions at inference. Reduces variance.

---

## The Visualization Toolkit — Use These Every Iteration

Never iterate without looking at these first. Numbers alone don't tell you why something failed.

| Figure | When to look | What to look for |
|---|---|---|
| `v1_overlay.png` | Every iteration | Where YOU drove vs where the NN drove. If they diverge, your training data is thin in that area |
| `v1_paths.png` | Every iteration | Where does the bot consistently crash or get stuck? |
| `v1_progress.png` | Every iteration | High variance = inconsistent model (lucky sometimes, bad others) |
| `fig_loss_v1.png` | After training | Val loss rising while train loss falls = overfitting. Stop training earlier or add data |
| `fig_actions_v1.png` | After collecting data | Are your turns symmetric? If you only turned right, the bot can't turn left |
| `fig_heading_v1.png` | After collecting data | Should slope downward. If it's flat/random, the network can't learn to navigate |
| `_history.png` | Every few iterations | Run `04_compare.py v1 v2 v3...` to see all iterations side by side |

---

## Commit Message Format — This is Half Your Grade

**Good format:**
```
v4-dagger: hand-corrected 600 frames after watching v3 stall against walls
on seed 42. Predicted: completion 60%→80%. Actual: 60%→80% on seed 42,
20%→60% on seed 7 (was overfit to seed 42 terrain).
```

**Bad format:**
```
updated stuff
```

Every commit must include:
1. The tag name at the start
2. What you changed and why
3. What you predicted would happen
4. What actually happened (including surprises)

---

## Required Files Per Iteration

For each iteration commit, you must include:

| File | When to include |
|---|---|
| `data_<tag>.npz` | Only if you collected new data |
| `nav_<tag>.npz` | Always |
| `benchmarks/<tag>.json` | Always |
| `benchmarks/<tag>_paths.png` | Always |
| `benchmarks/<tag>_progress.png` | Always |
| `benchmarks/<tag>_overlay.png` | When you have new data (pass `--data`) |
| `fig_loss_<tag>.png` | Always |
| `fig_actions_<tag>.png` | When you have new data |
| `fig_heading_<tag>.png` | When you have new data |

---

## Tournament Preparation Checklist

Before the final day, verify all of these:

- [ ] Benchmarked on **at least 3 different seeds** (42, 7, 99 minimum)
- [ ] Bot completes **at least one full lap** on multiple seeds (this is the pass/fail bar)
- [ ] Dataset includes **obstacle avoidance** driving (2 out of 5 rounds have obstacles)
- [ ] Dataset includes **wall recovery** driving
- [ ] Have 6–10 iterations with proper commit messages
- [ ] `_history.png` shows a visible improvement trend
- [ ] `benchmarks/` folder has `.json` + PNGs for every iteration
- [ ] Never edited `benchmark.py`
- [ ] All training data came from your own driving

---

## House Rules (Violations Can Affect Grade)

- Your driving data = your recordings only. No sharing with classmates.
- No external pretrained models. PyTorch from scratch is fine.
- Do not edit `benchmark.py`.
- One commit per iteration minimum (even bad results are still commits — that's your process).

---

## Summary — What a High-Grade Submission Looks Like

```
v1-bc         baseline, 2/5 completion
v2-recovery   added wall recovery data, 3/5 completion
v3-seeds      discovered seed 7 failure, planned fix
v4-deepnet    12→128→64→32→2 network, 4/5 completion
v5-dagger     hand-corrected 600 frames, 4/5→5/5 on seed 42
v6-smooth     added action smoothing, reduced crashes
v7-obstacles  recorded obstacle data, now passing obstacle rounds
v8-ensemble   averaged two model seeds, consistent 5/5
```

Each with a full commit message. Visible upward trend. Multiple seeds tested. Plots attached.

---

> The single most important thing to understand: **the grade is for showing your thinking process, not just the final result.** A bot that goes from 1/5 to 5/5 over 8 well-documented iterations beats a lucky 5/5 model with 2 vague commits.
