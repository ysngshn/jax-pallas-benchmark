from collections.abc import Callable, Sequence
from functools import partial
import jax
from jax.experimental import pallas as pl
from jax import Array, ShapeDtypeStruct as ArraySDS
from jax.lax import fori_loop, scan


# util functions


def _positive_dim(dim: int, ndim: int) -> int:
    return ndim + dim if dim < 0 else dim


def _ceil_div(x: int, y: int) -> int:
    return (x - 1) // y + 1


# pallas kernel


def _indrnn_elementwise_kernel(
        x_ref,
        whh_ref,
        h0_ref,
        # outputs
        o_ref,
        *,
        activation,
        seq_dim,
        channel_dim,
):
    seqlen = x_ref.shape[seq_dim]
    slice_all = tuple(slice(None) for _ in range(x_ref.ndim))
    whh_reshape = [1 if i != channel_dim else -1 for i in range(x.ndim)]
    whh_reshape.pop(seq_dim)
    whh = whh_ref[...].reshape(whh_reshape)
    h0 = h0_ref[...]

    def _step(i, h_prev):
        idx = slice_all[:seq_dim] + (i,) + slice_all[seq_dim+1:]
        h_next = activation(x_ref[idx] + whh * h_prev)
        o_ref[idx] = h_next
        return h_next

    _ = fori_loop(0, seqlen, _step, h0)


# indrnn implementations


def indrnn_naive(
    x: Array,
    whh: Array,
    h0: Array | None = None,
    activation: Callable[[Array], Array] = jax.nn.relu,
    seq_dim: int = -2,
    channel_dim: int = -1,
) -> Array:
    assert whh.ndim == 1
    assert x.shape[channel_dim] == whh.size
    ndim = x.ndim
    seq_dim = seq_dim + ndim if seq_dim < 0 else seq_dim
    channel_dim = channel_dim + ndim if channel_dim < 0 else channel_dim
    assert seq_dim != channel_dim

    whh_reshape = [1 for _ in range(x.ndim)]
    whh_reshape[channel_dim] = x.shape[channel_dim]
    whh_reshape.pop(seq_dim)
    whh = whh.reshape(whh_reshape)
    h_shape = list(x.shape)
    h_shape.pop(seq_dim)
    h = jax.numpy.zeros(h_shape, dtype=x.dtype) if h0 is None else h0
    outs = []
    idx = [slice(None)] * x.ndim  # type: list[slice | int]

    for i in range(x.shape[seq_dim]):
        idx[seq_dim] = i
        x_curr = x[tuple(idx)]
        h = activation(x_curr + whh * h)
        outs.append(h)

    return jnp.stack(outs, seq_dim)


def indrnn_scan(
    x: Array,
    whh: Array,
    h0: Array | None = None,
    activation: Callable[[Array], Array] = jax.nn.relu,
    seq_dim: int = -2,
    channel_dim: int = -1,
) -> Array:
    assert whh.ndim == 1
    assert x.shape[channel_dim] == whh.size
    ndim = x.ndim
    seq_dim = _positive_dim(seq_dim, ndim)
    channel_dim = _positive_dim(channel_dim, ndim)
    assert seq_dim != channel_dim

    whh_reshape = [1 for _ in range(x.ndim)]
    whh_reshape[channel_dim] = x.shape[channel_dim]
    whh_reshape.pop(seq_dim)
    whh = whh.reshape(whh_reshape)

    def _step(h_prev, x_curr):
        h_next = activation(x_curr + whh * h_prev)
        return h_next, h_next

    h_shape = list(x.shape)
    h_shape.pop(seq_dim)
    h0 = jax.numpy.zeros(h_shape, dtype=x.dtype) if h0 is None else h0

    x = jnp.moveaxis(x, seq_dim, 0)
    _, y = scan(_step, h0, x)
    return jnp.moveaxis(y, 0, seq_dim)


def indrnn_pallas(
    x: Array,
    whh: Array,
    h0: Array | None = None,
    activation: Callable[[Array], Array] = jax.nn.relu,
    seq_dim: int = -2,
    channel_dim: int = -1,
    block_shape: Sequence[int] | None = None,
    interpret: bool = False,
    debug: bool = False,
) -> Array:
    assert whh.ndim == 1
    assert x.shape[channel_dim] == whh.size
    ndim = x.ndim
    seq_dim = _positive_dim(seq_dim, ndim)
    channel_dim = _positive_dim(channel_dim, ndim)
    # get default (smallest) block shape if not specified
    if block_shape is None:
        block_shape = tuple(
            1 if i != seq_dim else x.shape[seq_dim] for i in range(x.ndim)
        )
    # get grid size
    grid = tuple(_ceil_div(ts, bs) for ts, bs in zip(x.shape, block_shape))
    # shouldn't cut along sequence dimension
    assert grid[seq_dim] == 1
    # make default h0 if needed
    if h0 is None:
        h_shape = list(x.shape)
        h_shape.pop(seq_dim)
        h0 = jax.numpy.zeros(h_shape, dtype=x.dtype)
    # block specs
    x_spec = pl.BlockSpec(block_shape, lambda *idx: idx)
    whh_spec = pl.BlockSpec(
        (block_shape[channel_dim],),
        lambda *idx: (idx[channel_dim],),
    )
    h_spec = pl.BlockSpec(
        block_shape[:seq_dim] + block_shape[seq_dim+1:],
        lambda *idx: idx[:seq_dim] + idx[seq_dim+1:],
    )
    return pl.pallas_call(
        partial(
            _indrnn_elementwise_kernel,
            activation=activation,
            seq_dim=seq_dim,
            channel_dim=channel_dim,
        ),
        out_shape=ArraySDS(x.shape, x.dtype),
        grid=grid,
        in_specs=[x_spec, whh_spec, h_spec],
        out_specs=x_spec,
        debug=debug,
        interpret=interpret,
    )(x, whh, h0)


if __name__ == "__main__":
    import jax.numpy as jnp
    import jax.random as jrand
    import timeit

    block_size = 8
    nt = 1000
    nc = 1000

    key = jrand.key(42)
    x = jax.random.normal(key, [nt, nc])
    whh = jnp.ones([nc])
    
    # check consistency
    y = indrnn_naive(x, whh)
    print("naive\n", y)
    indrnn_naive_jitted = jax.jit(indrnn_naive)
    y = indrnn_naive_jitted(x, whh)
    print("jitted\n", y)
    y = indrnn_scan(x, whh)
    print("scan\n", y)
    indrnn_scan_jitted = jax.jit(indrnn_scan)
    y = indrnn_scan_jitted(x, whh)
    print("scjit\n", y)
    y = indrnn_pallas(x, whh)
    print("pallas\n", y)
    indrnn_pallas_jitted = jax.jit(indrnn_pallas)
    y = indrnn_pallas_jitted(x, whh)
    print("paljit\n", y)
    indrnn_blocked = partial(indrnn_pallas, block_shape=[nt, block_size])
    y = indrnn_blocked(x, whh)
    print(f"block {block_size}\n", y)
    indrnn_blocked_jitted = jax.jit(indrnn_blocked)
    y = indrnn_blocked_jitted(x, whh)
    print(f"{block_size}jit\n", y)

    # benchmark naive v.s. scan v.s. pallas
    print("\n\n### estimated time for 10000 runs ###")
    t = timeit.timeit(
        'z = indrnn_naive(y, whh); z.block_until_ready()',
        number=1,
        globals=globals(),
    )
    print("naive\t", t * 1000)
    t = timeit.timeit(
        'z = indrnn_scan(y, whh); z.block_until_ready()',
        number=1,
        globals=globals(),
    )
    print("scan\t", t * 1000)
    t = timeit.timeit(
        'z = indrnn_pallas(y, whh); z.block_until_ready()',
        number=1,
        globals=globals(),
    )
    print("pallas\t", t * 1000)
    t = timeit.timeit(
        'z = indrnn_blocked(y, whh); z.block_until_ready()',
        number=1,
        globals=globals(),
    )
    print(f"b {block_size}\t", t * 1000)
    t = timeit.timeit(
        'z = indrnn_naive_jitted(y, whh); z.block_until_ready()',
        number=1000,
        globals=globals(),
    )
    print("jitted\t", t)
    t = timeit.timeit(
        'z = indrnn_scan_jitted(y, whh); z.block_until_ready()',
        number=1000,
        globals=globals(),
    )
    print("scjit\t", t)
    t = timeit.timeit(
        'z = indrnn_pallas_jitted(y, whh); z.block_until_ready()',
        number=100,
        globals=globals(),
    )
    print("paljit\t", t * 10)
    t = timeit.timeit(
        'z = indrnn_blocked_jitted(y, whh); z.block_until_ready()',
        number=1000,
        globals=globals(),
    )
    print(f"{block_size}jit\t", t)
