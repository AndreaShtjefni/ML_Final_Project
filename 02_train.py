"""Step 2 — Inspect data, write backprop, train, save weights.

Run:  python 02_train.py --data data_v1.npz --tag v1

It loads the dataset from `--data`, saves diagnostic figures, runs the
gradient check on YOUR `my_backward()`, trains for 300 epochs (Adam,
batch 64, lr 1e-3, 90/10 train/val), and saves nav_<tag>.npz.

The function `my_backward` near the top is yours to fill in. The script
asserts that your gradients agree with numerical_gradient before it lets
training start. If the assertion fires, fix the bug.

This script is the baseline. Once you've passed the gradient check and got
your first benchmark, the iteration loop is yours: change the architecture
in `drive2win/nn.py`, change the data, change the training
schedule, retrain, rebenchmark, commit, repeat.
"""
from __future__ import annotations
import argparse
import numpy as np

from drive2win import nn as nn_mod
from drive2win import viz
from drive2win.normalize import (
    normalize_states, FEATURE_NAMES, N_FEATURES, N_ACTIONS,
)


# =========================================================================
# TODO — write backward()
# =========================================================================
# Walk the chain rule outward from the loss:
#   y = tanh(z3),  loss = MSE(y, target)
#   z3 = a2 W3 + b3,   a2 = ReLU(z2)
#   z2 = a1 W2 + b2,   a1 = ReLU(z1)
#   z1 = x  W1 + b1
#
# Replace each `...` with the correct expression.
# =========================================================================
def my_backward(x, y_target, w, cache):
    # cast to float64 so gradient check passes regardless of NumPy version
    x        = x.astype(np.float64)
    y_target = y_target.astype(np.float64)
    cache    = {k: v.astype(np.float64) for k, v in cache.items()}
    w        = {k: v.astype(np.float64) for k, v in w.items()}

    n = x.shape[0]
    y = cache["y"]
    # --- output ---
    dy  = 2.0 * (y - y_target) / (n * y.shape[1])  # MSE gradient
    dz3 = dy * (1.0 - y * y)                        # tanh derivative: 1 - tanh²
    dW3 = cache["a2"].T @ dz3                       # gradient for W3
    db3 = dz3.sum(axis=0)                           # gradient for b3
    # --- hidden 2 ---
    da2 = dz3 @ w["W3"].T                           # backprop through W3
    dz2 = da2 * (cache["z2"] > 0)                   # ReLU mask: zero where z2 was ≤ 0
    dW2 = cache["a1"].T @ dz2                       # gradient for W2
    db2 = dz2.sum(axis=0)                           # gradient for b2
    # --- hidden 1 ---
    da1 = dz2 @ w["W2"].T                           # backprop through W2
    dz1 = da1 * (cache["z1"] > 0)                   # ReLU mask: zero where z1 was ≤ 0
    dW1 = x.T @ dz1                                 # gradient for W1
    db1 = dz1.sum(axis=0)                           # gradient for b1
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def gradient_check():
    rng = np.random.default_rng(0)
    w = nn_mod.init_weights(seed=0)
    # use float64 so numerical and analytical gradients are at the same precision
    w  = {k: v.astype(np.float64) for k, v in w.items()}
    x = rng.normal(size=(8, N_FEATURES)).astype(np.float64)
    y = rng.uniform(-1, 1, size=(8, N_ACTIONS)).astype(np.float64)
    cache = nn_mod.forward_all(x, w)
    grads = my_backward(x, y, w, cache)

    print("\ngradient check (max relative error per parameter):")
    for key in w:
        max_err = 0.0
        flat = w[key].size
        for _ in range(5):
            idx = np.unravel_index(rng.integers(0, flat), w[key].shape)
            num = nn_mod.numerical_gradient(x, y, w, key, idx)
            ana = grads[key][idx]
            denom = max(1e-6, abs(num) + abs(ana))
            max_err = max(max_err, abs(num - ana) / denom)
        flag = "OK" if max_err < 1e-4 else "BUG"
        print(f"  {key}: {max_err:.2e}   {flag}")
        assert max_err < 1e-4, (
            f"backward() gradient for {key} disagrees with numerical_gradient. "
            f"Fix it before training."
        )


def inspect_dataset(states_raw, actions, tag: str):
    print("\nfeature ranges (raw):")
    for i, name in enumerate(FEATURE_NAMES):
        col = states_raw[:, i]
        print(f"  {name:>20s}: [{col.min():+7.2f}, {col.max():+7.2f}]   "
              f"mean={col.mean():+.2f}  std={col.std():.2f}")
    viz.plot_action_histograms(actions, out=f"fig_actions_{tag}.png")
    viz.plot_heading_vs_steering(states_raw, actions, out=f"fig_heading_{tag}.png")


def train(X, Y, epochs=300, lr=1e-3, batch_size=64, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    N = len(X)

    # Stratified train/val split — ensures validation covers all track zones,
    # not just early checkpoints. Splits by front-ray quartile as a proxy for
    # track position (open road vs tight obstacle zones).
    quartiles = np.percentile(X[:, 2], [25, 50, 75])
    strata = np.digitize(X[:, 2], quartiles)
    val_idx = []
    for s in np.unique(strata):
        s_idx = np.where(strata == s)[0]
        rng_s = np.random.default_rng(seed + int(s))
        rng_s.shuffle(s_idx)
        n_take = max(1, int(len(s_idx) * val_frac))
        val_idx.extend(s_idx[:n_take].tolist())
    val_idx = np.array(val_idx)
    tr_idx = np.setdiff1d(np.arange(N), val_idx)
    Xtr, Ytr, Xva, Yva = X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]

    # Left/right flip augmentation — doubles training data for free.
    # A car turning right with rays [A,B,C,D,E,F,G,H] should turn left at the
    # same magnitude when the car is mirrored. Swap symmetric ray pairs and
    # negate heading_error + steering. Applied to training only, never val.
    # Column layout after checkpoint_distance drop:
    #   0=speed, 1=heading_error, 2=front, 3=+45, 4=+90, 5=+135,
    #   6=back, 7=-135, 8=-90, 9=-45, 10=friction
    Xflip = Xtr.copy()
    Xflip[:, 1] = -Xtr[:, 1]                                        # negate heading_error
    Xflip[:, 3], Xflip[:, 9] = Xtr[:, 9].copy(), Xtr[:, 3].copy()  # +45 <-> -45
    Xflip[:, 4], Xflip[:, 8] = Xtr[:, 8].copy(), Xtr[:, 4].copy()  # +90 <-> -90
    Xflip[:, 5], Xflip[:, 7] = Xtr[:, 7].copy(), Xtr[:, 5].copy()  # +135 <-> -135
    Yflip = Ytr.copy()
    Yflip[:, 1] = -Ytr[:, 1]                                        # negate steering
    Xtr = np.concatenate([Xtr, Xflip], axis=0)
    Ytr = np.concatenate([Ytr, Yflip], axis=0)

    # Obstacle zone oversampling — frames where front ray < 0.3 (normalized, ~15m)
    # are the exact failure point (cp2->cp3 obstacle cluster). Oversample 3x so the
    # network sees more examples of the hard-left arc maneuver during each epoch.
    obstacle_mask = Xtr[:, 2] < 0.3
    n_obstacle = obstacle_mask.sum()
    if n_obstacle > 0:
        Xtr = np.concatenate([Xtr, Xtr[obstacle_mask], Xtr[obstacle_mask]], axis=0)
        Ytr = np.concatenate([Ytr, Ytr[obstacle_mask], Ytr[obstacle_mask]], axis=0)
        print(f"obstacle oversampling: {n_obstacle} frames x3 added to training")

    w = nn_mod.init_weights(seed=seed)
    state = nn_mod.init_adam(w)
    train_losses, val_losses = [], []
    best_val = float("inf"); best = {k: v.copy() for k, v in w.items()}

    for epoch in range(epochs):
        idx = rng.permutation(len(Xtr))
        Xs, Ys = Xtr[idx], Ytr[idx]
        ep_loss, n_b = 0.0, 0
        for i in range(0, len(Xs), batch_size):
            xb, yb = Xs[i:i+batch_size], Ys[i:i+batch_size]
            cache = nn_mod.forward_all(xb, w)
            ep_loss += nn_mod.mse_loss(cache["y"], yb); n_b += 1
            grads = my_backward(xb, yb, w, cache)
            current_lr = lr if epoch < epochs // 2 else lr * 0.1
            nn_mod.adam_step(w, grads, state, lr=current_lr)
        v = nn_mod.mse_loss(nn_mod.forward(Xva, w), Yva)
        train_losses.append(ep_loss / max(1, n_b)); val_losses.append(v)
        if v < best_val:
            best_val = v; best = {k: w[k].copy() for k in w}
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  train={train_losses[-1]:.4f}  val={v:.4f}  best={best_val:.4f}")

    return best, train_losses, val_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v1.npz",
                    help="Dataset file from 01_collect.py")
    ap.add_argument("--tag", default="v1",
                    help="Output suffix (nav_<tag>.npz, fig_*_<tag>.png)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    states_raw, actions = d["states"], d["actions"]
    # Drop column 2 (checkpoint_distance) — it causes the model to freeze at
    # every checkpoint transition because the value jumps to a new direction
    # and distance the network has never seen. heading_error already encodes
    # which way to face for the next checkpoint, so nothing is lost.
    if states_raw.shape[1] == 12:
        states_raw = np.delete(states_raw, 2, axis=1)
        print("dropped checkpoint_distance column (was col 2) -> 11 features")
    print(f"raw states  : {states_raw.shape}")
    print(f"raw actions : {actions.shape}")

    inspect_dataset(states_raw, actions, tag=args.tag)

    X = normalize_states(states_raw)
    Y = actions.astype(np.float32)
    print(f"\nX range : [{X.min():+.2f}, {X.max():+.2f}]")
    print(f"Y range : [{Y.min():+.2f}, {Y.max():+.2f}]")

    gradient_check()

    weights, tr_losses, va_losses = train(
        X, Y, epochs=args.epochs, lr=args.lr, batch_size=args.batch)

    viz.plot_loss_curves(tr_losses, va_losses, out=f"fig_loss_{args.tag}.png")
    nn_mod.save(weights, f"nav_{args.tag}.npz")
    print(f"Saved nav_{args.tag}.npz")


if __name__ == "__main__":
    main()
