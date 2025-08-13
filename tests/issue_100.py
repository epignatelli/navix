import jax
import navix as nx


# test_perf_vision_pytest.py
import os
import time
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import lax


RUN_PERF = False


# --- Optimized crop (from before) ---
# @partial(jax.jit, static_argnames=("radius", "padding_value"))
def crop(
    grid: jnp.ndarray,
    origin: jnp.ndarray,
    direction: jnp.ndarray,
    radius: int,
    padding_value: int = 100,
) -> jnp.ndarray:
    origin = jnp.asarray(origin, jnp.int32)
    direction = jnp.asarray(direction, jnp.int32)

    pad_spec = ((radius, radius, 0), (radius, radius, 0)) + ((0, 0, 0),) * (
        grid.ndim - 2
    )
    x = lax.pad(grid, jnp.asarray(padding_value, grid.dtype), pad_spec)

    size = 2 * radius + 1
    x = lax.dynamic_slice_in_dim(x, origin[0], size, axis=0)
    x = lax.dynamic_slice_in_dim(x, origin[1], size, axis=1)

    def rot90(z):
        return jnp.flip(jnp.swapaxes(z, 0, 1), axis=0)

    def rot180(z):
        return jnp.flip(jnp.flip(z, axis=0), axis=1)

    def rot270(z):
        return jnp.flip(jnp.swapaxes(z, 0, 1), axis=1)

    def rot0(z):
        return z

    x = lax.switch(direction, (rot90, rot180, rot270, rot0), x)

    left = radius // 2
    right = 2 * radius - left + 1
    x = x[: radius + 1, left:right]
    return x.astype(grid.dtype)


# --- Optimized view_cone ---
# @partial(jax.jit, static_argnames=("radius",))
def view_cone(
    transparency_map: jnp.ndarray, origin: jnp.ndarray, radius: int
) -> jnp.ndarray:
    """Visibility with 3x3 dilation iterated `radius` times (bounded grid)."""
    origin = jnp.asarray(origin, jnp.int32)
    transp = transparency_map != 0  # boolean transparency

    v = jnp.zeros_like(transp, dtype=jnp.bool_)
    v = v.at[origin[0], origin[1]].set(True)

    # float32 conv to satisfy CuDNN
    k = jnp.ones((3, 3, 1, 1), dtype=jnp.float32)

    def step(_, vv):
        # CUDA does not support conv in int8
        vin = vv.astype(jnp.float32)[None, ..., None]  # (1,H,W,1)
        y = lax.conv_general_dilated(
            vin,
            k,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )[0, ..., 0]
        nxt = (y > 0.0) & transp  # stop propagation at opaque
        return nxt

    v = lax.fori_loop(0, radius, lambda i, vv: step(i, vv), v)
    v = v | (~transp)  # include opaque tiles as visible boundary
    return v.astype(transparency_map.dtype)


# -------------------------
# Helpers
# -------------------------
def _block(x):
    jax.tree_util.tree_map(
        lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x
    )


def time_steady_state(fn, *args, iters: int = 100) -> float:
    _block(fn(*args))  # warmup (includes compile)
    start = time.perf_counter()
    for _ in range(iters):
        _block(fn(*args))
    end = time.perf_counter()
    return (end - start) / iters


def make_rng_ints(key, shape, low, high, dtype=jnp.int32):
    return jax.random.randint(key, shape, low, high, dtype=dtype)


# -------------------------
# Fixtures
# -------------------------
@pytest.fixture(scope="module")
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def params():
    return dict(H=256, W=256, radius=24, padding_value=100, direction=3)


@pytest.fixture(scope="module")
def grid(rng, params):
    return make_rng_ints(
        jax.random.split(rng, 1)[0], (params["H"], params["W"]), 0, 256, jnp.int32
    )


@pytest.fixture(scope="module")
def transparency(rng, params):
    k = jax.random.split(rng, 2)[1]
    return (make_rng_ints(k, (params["H"], params["W"]), 0, 4) > 0).astype(jnp.int32)


@pytest.fixture(scope="module")
def origin(params):
    return jnp.array([params["H"] // 2, params["W"] // 2], dtype=jnp.int32)


# -------------------------
# Correctness / shape tests (always run)
# -------------------------
def test_crop_shapes_and_dtype(grid, origin, params):
    r = params["radius"]
    pad = params["padding_value"]
    direction = jnp.array(params["direction"], dtype=jnp.int32)

    out = crop(grid, origin, direction, r, pad)
    _block(out)

    # rows
    assert out.shape[0] == r + 1
    # columns (depends on parity)
    left = r // 2
    expected_w = 2 * r - 2 * left + 1
    assert out.shape[1] == expected_w
    assert out.dtype == grid.dtype


def test_view_cone_shapes_and_dtype(transparency, origin, params):
    r = params["radius"]
    out = view_cone(transparency, origin, r)
    _block(out)
    assert out.shape == transparency.shape
    assert out.dtype == transparency.dtype
