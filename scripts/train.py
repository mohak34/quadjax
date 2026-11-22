"""Train a policy with PPO on one task, test it, and save the learning curve and weights.

Usage: train.py hover|waypoint|forest|failure
"""

import pickle
import sys
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from quadjax import failure, forest, hover, ppo, waypoint

OUT = Path("out")


def train[S](
    name: str,
    env: ppo.Env[S],
    cfg: ppo.Config,
    max_steps: int,
    checks: list[tuple[str, int]],
    distance: Callable[[S], jax.Array],
) -> None:
    """distance(state) is one drone's true distance to its target, used by the test."""
    init, update = ppo.make_train(env, cfg)

    print(f"task: {name}  device: {jax.devices()[0]}  updates: {cfg.num_updates}")
    print(
        f"best possible episode return is {max_steps} (on target and still every step)"
    )
    runner = init(jax.random.key(0))
    rows = []
    t0 = time.perf_counter()
    for i in range(cfg.num_updates):
        runner, m = update(runner)
        steps = (i + 1) * cfg.num_envs * cfg.num_steps
        rows.append((steps, float(m["episode_return"])))
        if i % 10 == 0 or i == cfg.num_updates - 1:
            print(
                f"{steps / 1e6:6.1f}M steps  return {rows[-1][1]:6.1f}  "
                f"noise {float(m['noise_std']):.3f}  {time.perf_counter() - t0:5.0f} s"
            )

    OUT.mkdir(exist_ok=True)
    np.savetxt(
        OUT / f"{name}_curve.csv",
        rows,
        delimiter=",",
        header="steps,episode_return",
        comments="",
    )
    with open(OUT / f"{name}_params.pkl", "wb") as f:
        pickle.dump(jax.device_get(runner.params), f)

    # Test: 1000 fresh episodes, policy without exploration noise.
    def test_step(
        carry: tuple[S, jax.Array], _step: None
    ) -> tuple[tuple[S, jax.Array], tuple[jax.Array, jax.Array]]:
        s, obs = carry
        s, obs, _, crashed, _ = jax.vmap(env.step)(s, ppo.mlp(runner.params.actor, obs))
        return (s, obs), (jax.vmap(distance)(s), crashed)

    start = jax.vmap(env.reset)(jax.random.split(jax.random.key(1), 1000))
    _, (dist, crashed) = jax.lax.scan(test_step, start, None, max_steps)
    crashed = crashed.any(axis=0)
    ok = dist[:, ~crashed]
    print(f"\ntest on 1000 new starts: crashed {int(crashed.sum())}")
    for label, t in checks if ok.shape[1] else []:
        d = ok[t]
        print(
            f"  {label}: median distance {float(jnp.median(d)) * 100:5.1f} cm, worst {float(d.max()) * 100:5.1f} cm"
        )

    plt.rcParams.update(
        {
            "figure.facecolor": "#000",
            "axes.facecolor": "#000",
            "axes.edgecolor": "#55554f",
            "text.color": "#f0efec",
            "axes.labelcolor": "#f0efec",
            "xtick.color": "#f0efec",
            "ytick.color": "#f0efec",
            "grid.color": "#222",
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    steps, ret = np.array(rows).T
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(steps / 1e6, ret, color="#3987e5", lw=2)
    ax.axhline(max_steps, color="#55554f", ls=(0, (4, 3)))
    ax.text(0, max_steps * 1.01, "max possible", color="#c3c2b7", va="bottom")
    ax.set_title(f"{name}: average episode return", loc="left")
    ax.set_xlabel("environment steps (millions)")
    fig.savefig(OUT / f"{name}_curve.png", dpi=110)
    print(f"saved {OUT}/{name}_curve.png, {name}_curve.csv, {name}_params.pkl")


match sys.argv[1:]:
    case ["hover"]:
        h = hover
        train(
            "hover",
            ppo.Env(h.reset, h.step, h.OBS_SIZE, h.ACT_SIZE),
            h.CONFIG,
            h.MAX_STEPS,
            h.CHECKS,
            h.distance,
        )
    case ["waypoint"]:
        w = waypoint
        train(
            "waypoint",
            ppo.Env(w.reset, w.step, w.OBS_SIZE, w.ACT_SIZE),
            w.CONFIG,
            w.MAX_STEPS,
            w.CHECKS,
            w.distance,
        )
    case ["forest"]:
        f = forest
        train(
            "forest",
            ppo.Env(f.reset, f.step, f.OBS_SIZE, f.ACT_SIZE),
            f.CONFIG,
            f.MAX_STEPS,
            f.CHECKS,
            f.distance,
        )
    case ["failure"]:
        m = failure
        train(
            "failure",
            ppo.Env(m.reset, m.step, m.OBS_SIZE, m.ACT_SIZE),
            m.CONFIG,
            m.MAX_STEPS,
            m.CHECKS,
            m.distance,
        )
    case _:
        sys.exit("usage: train.py hover|waypoint|forest|failure")
