"""Train a hover policy with PPO, then test it and save the learning curve and weights."""

import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from volatus import hover, ppo

OUT = Path("out")
env = ppo.Env(hover.reset, hover.step, hover.OBS_SIZE, hover.ACT_SIZE)
cfg = ppo.Config()
init, update = ppo.make_train(env, cfg)

print(f"device: {jax.devices()[0]}  envs: {cfg.num_envs}  updates: {cfg.num_updates}")
print(
    f"best possible episode return is {hover.MAX_STEPS} (on target, still, for all 5 s)"
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
    OUT / "hover_curve.csv",
    rows,
    delimiter=",",
    header="steps,episode_return",
    comments="",
)
with open(OUT / "hover_params.pkl", "wb") as f:
    pickle.dump(jax.device_get(runner.params), f)


# Test: 1000 fresh drones, policy without exploration noise, one full episode each.
def test_step(
    carry: tuple[hover.EnvState, jax.Array], _step: None
) -> tuple[tuple[hover.EnvState, jax.Array], tuple[jax.Array, jax.Array]]:
    s, obs = carry
    s, obs, _, crashed, _ = jax.vmap(hover.step)(s, ppo.mlp(runner.params.actor, obs))
    return (s, obs), (jnp.linalg.norm(obs[:, :3], axis=1), crashed)


start = jax.vmap(hover.reset)(jax.random.split(jax.random.key(1), 1000))
_, (dist, crashed) = jax.lax.scan(test_step, start, None, hover.MAX_STEPS)
crashed = crashed.any(axis=0)
ok = dist[:, ~crashed]
print(f"\ntest on 1000 new starts: crashed {int(crashed.sum())}")
for sec in (1, 2, 5):
    d = ok[sec * 50 - 1]
    print(
        f"  after {sec} s: median distance to target {float(jnp.median(d)) * 100:5.1f} cm, "
        f"worst {float(d.max()) * 100:5.1f} cm"
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
ax.axhline(hover.MAX_STEPS, color="#55554f", ls=(0, (4, 3)))
ax.text(0, hover.MAX_STEPS + 3, "max possible", color="#c3c2b7", va="bottom")
ax.set_title("hover: average episode return", loc="left")
ax.set_xlabel("environment steps (millions)")
fig.savefig(OUT / "hover_curve.png", dpi=110)
print(f"saved {OUT}/hover_curve.png, hover_curve.csv, hover_params.pkl")
