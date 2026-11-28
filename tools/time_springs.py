"""Time the spring kernels (update, residual, stiffness blocks) on one device.

    uv run tools/time_springs.py <device> [count ...]

Best of 7 after a warm-up. On CUDA each timing ends with wp.synchronize_device.
`update` is timed twice: the kernel alone, and Term.update as the run path calls
it, which also reads the status code back to the host.
"""

import sys
import time

import numpy as np
import warp as wp

from warp_febio.assembly.system import State
from warp_febio.elements.discrete_domain import DiscreteDomain
from warp_febio.materials.models.nonlinear_spring import nonlinear_spring

wp.config.quiet = True
device = sys.argv[1]
counts = [int(c) for c in sys.argv[2:]] or [100, 1_000_000]
CURVES = [[60.0, 0.0, 0.0, 0.4, 0.0, 1.4, 1.0], [30.0, 0.0, 0.0, 0.2, 0.0, 1.2, 1.0]]


def best(fn, repeats=7):
    fn()
    times = []
    for _ in range(repeats):
        wp.synchronize_device(device)
        t = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        times.append(time.perf_counter() - t)
    return 1e3 * min(times)


for n in counts:
    rng = np.random.default_rng(0)
    N = n + 3
    X = rng.uniform(-3.0, 3.0, (N, 3))
    u = rng.uniform(-0.3, 0.3, (N, 3))
    a = rng.integers(0, N, n)
    b = (a + 1 + rng.integers(0, N - 1, n)) % N
    pairs = np.stack([a, b], axis=1).astype(np.int32)
    term = DiscreteDomain(
        "springs",
        nonlinear_spring(3, "elongation", "extrapolate"),
        X[pairs],
        pairs,
        CURVES,
        N,
        device,
        region=np.arange(n) % 2,
    )
    blocks = term.block_count
    term.bind_blocks(
        wp.zeros(blocks, dtype=wp.int32, device=device),
        wp.zeros(blocks, dtype=wp.int32, device=device),
        wp.zeros(blocks, dtype=wp.mat33d, device=device),
    )
    state = State(0.0, wp.array(u, dtype=wp.vec3d, device=device))
    R = wp.zeros(N, dtype=wp.vec3d, device=device)
    assert term.update(state) == 0
    tension = int((term.data["axial_force"].numpy() > 0).sum())
    row = dict(
        update_kernel=best(lambda: term._launch_update(state, term.data, term.status)),
        update_with_status=best(lambda: term.update(state)),
        residual=best(lambda: term.residual(state, None, R)),
        stiffness=best(lambda: term.linearize(state, None)),
    )
    print(f"{device} n={n} in_tension={tension} " + " ".join(f"{k}={v:.3f}ms" for k, v in row.items()))
