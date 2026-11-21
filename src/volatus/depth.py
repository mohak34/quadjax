"""A fake depth camera: shoot a grid of rays out of the drone's nose and measure how far each
one goes before it hits a tree trunk or the ground.

Trunks are vertical cylinders of endless height, so a ray only needs a 2D circle test.
The camera is a pinhole looking along the drone's forward axis, fixed to the body:
when the drone pitches nose-down to fly, the picture tilts toward the ground with it.
"""

import jax
import jax.numpy as jnp

from volatus import sim

WIDTH, HEIGHT = 32, 8  # pixels
HFOV, VFOV = 90.0, 60.0  # degrees, field of view across and up-down
MAX_DEPTH = 6.0  # m, anything farther reads as this
PIXELS = WIDTH * HEIGHT


def _pinhole_rays() -> jax.Array:
    """(PIXELS, 3) unit directions in the body frame, row by row from the top-left pixel.
    Body y is left, so the left column looks toward +y."""
    left = jnp.tan(jnp.radians(HFOV / 2)) * jnp.linspace(1, -1, WIDTH)
    up = jnp.tan(jnp.radians(VFOV / 2)) * jnp.linspace(1, -1, HEIGHT)
    y, z = jnp.meshgrid(left, up)  # (HEIGHT, WIDTH) each
    d = jnp.stack([jnp.ones_like(y), y, z], -1).reshape(-1, 3)
    return d / jnp.linalg.norm(d, axis=1, keepdims=True)


RAYS = _pinhole_rays()


def cast(origin: jax.Array, dirs: jax.Array, trees: jax.Array) -> jax.Array:
    """Distance along each unit ray in dirs (r, 3) from origin (3,) to the first tree or the
    ground, capped at MAX_DEPTH. trees is (n, 3): center x, center y, radius."""
    # Point on ray: o + t d. It is on a trunk when its (x, y) is radius away from the center:
    # |o2 + t d2|^2 = r^2, a quadratic a t^2 + 2 b t + c = 0. The smaller root is where it enters.
    o2 = origin[:2] - trees[:, :2]  # (n, 2)
    d2 = dirs[:, :2]  # (r, 2)
    # a is ~0 only for a ray pointing straight up or down, which never meets a trunk.
    a = jnp.maximum((d2 * d2).sum(-1, keepdims=True), 1e-9)  # (r, 1)
    b = jnp.dot(d2, o2.T, precision="highest")  # (r, n)
    c = (o2 * o2).sum(-1) - trees[:, 2] ** 2  # (n,)
    disc = b * b - a * c
    t = (-b - jnp.sqrt(jnp.maximum(disc, 0.0))) / a
    t_tree = jnp.where((disc > 0) & (t > 0), t, jnp.inf).min(axis=1)

    dz = jnp.minimum(dirs[:, 2], -1e-6)  # only rays pointing down reach the ground
    t_ground = jnp.where(dirs[:, 2] < -1e-6, -origin[2] / dz, jnp.inf)
    return jnp.minimum(jnp.minimum(t_tree, t_ground), MAX_DEPTH)


def image(pos: jax.Array, quat: jax.Array, trees: jax.Array) -> jax.Array:
    """(PIXELS,) depth in m seen by a drone at pos with orientation quat."""
    # Full float32 matmul: the GPU default (TF32) is only good to ~1 cm at these distances.
    return cast(pos, jnp.dot(RAYS, sim.rotation(quat).T, precision="highest"), trees)
