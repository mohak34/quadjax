"""How fast the sim runs: many drones stepped together on the GPU."""

import time

import jax
import jax.numpy as jnp

from quadjax import sim

STEPS = 1000  # 2 s of flight at 500 Hz


def run(n: int) -> None:
    start = jax.vmap(sim.hover_state)(jnp.zeros((n, 3)))
    cmd = jnp.full((n, 4), sim.HOVER_THRUST)
    step_all = jax.vmap(sim.step)

    @jax.jit
    def fly(s: sim.State) -> sim.State:
        return jax.lax.scan(lambda s, _: (step_all(s, cmd), None), s, None, STEPS)[0]

    fly(start).pos.block_until_ready()  # first call compiles; don't time it
    t0 = time.perf_counter()
    fly(start).pos.block_until_ready()
    secs = time.perf_counter() - t0

    steps_per_sec = n * STEPS / secs
    print(
        f"{n:>7} drones  {secs * 1000:8.1f} ms  {steps_per_sec / 1e6:8.1f} M steps/s  "
        f"{steps_per_sec * sim.DT:10.0f} flight-seconds per real second"
    )


print(f"device: {jax.devices()[0]}")
for n in (1, 1024, 4096, 16384, 65536):
    run(n)
