"""Run with python -m warp_febio_pilot.analysis.explicit_patch; no external solver dependencies."""

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import warp as wp
from warp_febio_pilot.mesh.stl import load_patch
from warp_febio_pilot.elements.triangular_membrane import assemble, advance, ramp_pressure


def reference_data(x, tri, thickness, density):
    """Precompute each triangle's 2D rest basis and lump mass onto its vertices.

    x: reference coordinates [m]; tri: vertex indices; thickness [m]; density [kg/m3].
    The inverse basis converts current edge vectors into the surface deformation
    gradient. The minimum altitude controls the explicit stable-step estimate.
    """
    a = x[tri[:, 1]] - x[tri[:, 0]]
    b = x[tri[:, 2]] - x[tri[:, 0]]
    length = np.linalg.norm(a, axis=1)
    area = np.linalg.norm(np.cross(a, b), axis=1) / 2
    dm = np.zeros((len(tri), 2, 2))
    dm[:, 0, 0] = length
    dm[:, 0, 1] = np.sum(a * b, axis=1) / length
    dm[:, 1, 1] = 2 * area / length
    mass = np.zeros(len(x))
    for j in range(3):
        np.add.at(mass, tri[:, j], density * thickness * area / 3)
    edges = np.stack(
        [
            np.linalg.norm(a, axis=1),
            np.linalg.norm(b, axis=1),
            np.linalg.norm(b - a, axis=1),
        ]
    )
    altitude = float(np.min(2 * area / edges.max(axis=0)))
    return np.linalg.inv(dm), area, mass, altitude


def main():
    start = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, default=Path("synthetic_patch.stl"))
    p.add_argument("--output", type=Path, default=Path("outputs/patch"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--duration", type=float, default=0.05)
    p.add_argument("--ramp", type=float, default=0.01)
    p.add_argument("--dt-scale", type=float, default=1.0)
    p.add_argument("--pressure-mmhg", type=float, default=40.0)
    p.add_argument(
        "--young-mpa",
        type=float,
        default=10.0,
        help="Synthetic benchmark modulus in MPa; not clinical ePTFE",
    )
    args = p.parse_args()
    if not np.isfinite(args.young_mpa) or args.young_mpa <= 0:
        p.error("--young-mpa must be finite and positive")
    if (
        not 0 < args.ramp <= args.duration
        or not 0 < args.dt_scale <= 1
        or args.pressure_mmhg < 0
    ):
        p.error("Require 0 < ramp <= duration, 0 < dt-scale <= 1, pressure >= 0")
    args.output.mkdir(parents=True, exist_ok=True)
    x0, tri, boundary, mesh_report = load_patch(args.mesh)
    # Center coordinates to reduce float32 cancellation; retain origin for export.
    origin = x0.mean(axis=0)
    x0 = x0 - origin
    young = args.young_mpa * 1.0e6
    nu = 0.3
    thickness = 0.0005
    density = 1000.0
    damping = 2000.0
    # Plane-stress Lamé coefficient: different from full 3D elasticity.
    lame = young * nu / (1 - nu**2)
    mu = young / (2 * (1 + nu))
    inv, area, mass, altitude = reference_data(x0, tri, thickness, density)
    wave = np.sqrt(young / (density * (1 - nu**2)))
    dt_limit = 0.15 * altitude / wave * args.dt_scale
    steps = int(np.ceil(args.duration / dt_limit))
    dt = args.duration / steps
    fixed = np.zeros(len(x0), np.int32)
    fixed[boundary] = 1
    free = fixed == 0
    wp.config.kernel_cache_dir = str(Path(".warp_cache").resolve())
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise ValueError("This benchmark requires CUDA; no CPU fallback")
    with wp.ScopedDevice(device):
        x = wp.array(x0, dtype=wp.vec3)
        rest = wp.clone(x)
        vel = wp.zeros_like(x)
        f = wp.zeros_like(x)
        pf = wp.zeros_like(x)
        faces = wp.array(tri, dtype=wp.vec3i)
        dinv = wp.array(inv, dtype=wp.mat22)
        ar = wp.array(area, dtype=float)
        m = wp.array(mass, dtype=float)
        bc = wp.array(fixed, dtype=int)
        pressure_device = wp.zeros(1, dtype=float)
        counter = wp.zeros(1, dtype=int)

        # Assemble internal plus follower-pressure force on the CURRENT triangles.
        def forces():
            f.zero_()
            pf.zero_()
            wp.launch(
                assemble,
                len(tri),
                [x, faces, dinv, ar, f, pf, thickness, lame, mu, pressure_device],
            )

        # Compile and warm all kernels without advancing a nonzero load.
        forces()
        wp.launch(advance, len(x0), [x, rest, vel, f, m, bc, 0.0, damping])
        wp.copy(x, rest)
        vel.zero_()
        wp.synchronize_device(device)

        def step_once():
            wp.launch(
                ramp_pressure,
                1,
                [
                    counter,
                    pressure_device,
                    dt,
                    args.ramp,
                    args.pressure_mmhg * 133.322387415,
                ],
            )
            forces()
            wp.launch(advance, len(x0), [x, rest, vel, f, m, bc, dt, damping])

        # Warm pressure kernel, then restore all state before graph capture.
        step_once()
        wp.copy(x, rest)
        vel.zero_()
        counter.zero_()
        pressure_device.zero_()
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(200):
                step_once()
        graph = capture.graph
        wp.synchronize_device(device)
        setup_seconds = time.perf_counter() - start
        history = []
        snapshots = [x0.astype(np.float32)]
        snapshot_times = [0.0]
        event_start = wp.Event(enable_timing=True)
        event_end = wp.Event(enable_timing=True)
        solve_start = time.perf_counter()
        gpu_ms = 0.0
        stride = max(1, steps // 100)
        # Advance in batches; host reads below are diagnostic/output work.
        for first in range(0, steps, stride):
            wp.record_event(event_start)
            count = min(stride, steps - first)
            for _ in range(count // 200):
                wp.capture_launch(graph)
            for _ in range(count % 200):
                step_once()
            fraction = min(1.0, (first + count) * dt / args.ramp)
            pressure = (
                args.pressure_mmhg
                * 133.322387415
                * fraction
                * fraction
                * (3 - 2 * fraction)
            )
            wp.record_event(event_end)
            gpu_ms += wp.get_event_elapsed_time(event_start, event_end)
            # Reassemble at the observed state for an undamped equilibrium residual.
            forces()
            xn = x.numpy()
            vn = vel.numpy()
            fn = f.numpy()
            pn = pf.numpy()
            if not all(np.isfinite(a).all() for a in (xn, vn, fn, pn)):
                raise RuntimeError("Nonfinite state: no result accepted")
            displacement = np.linalg.norm(xn - x0, axis=1)
            residual = np.linalg.norm(fn[free]) / max(np.linalg.norm(pn[free]), 1.0e-30)
            history.append(
                [
                    min(first + stride, steps) * dt,
                    pressure,
                    float(displacement.max() * 1.0e3),
                    float(residual),
                    float(np.linalg.norm(vn, axis=1).max()),
                    float(0.5 * np.sum(mass[:, None] * vn**2)),
                ]
            )
            snapshots.append(xn)
            snapshot_times.append(history[-1][0])
        wp.synchronize_device(device)
        solve_seconds = time.perf_counter() - solve_start
    if not np.array_equal(xn[boundary], x0.astype(np.float32)[boundary]):
        raise RuntimeError("Boundary moved")
    np.savez_compressed(
        args.output / "state.npz",
        rest=x0,
        positions=xn,
        faces=tri,
        boundary=boundary,
        coordinate_origin_m=origin,
        snapshots=np.array(snapshots),
        times=snapshot_times,
        velocity=vn,
        force=fn,
    )
    np.savetxt(
        args.output / "history.csv",
        history,
        delimiter=",",
        header="time_s,pressure_Pa,max_displacement_mm,relative_free_force_residual,max_speed_m_s,kinetic_energy_J",
        comments="",
    )
    result = dict(
        mesh=mesh_report,
        mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        material=dict(
            label="SYNTHETIC BENCHMARK ONLY — not clinical ePTFE",
            young_Pa=young,
            poisson=nu,
            thickness_m=thickness,
            density_kg_m3=density,
        ),
        formulation="finite-strain StVK plane-stress triangular membrane; no bending",
        integration=dict(
            method="damped semi-implicit Euler",
            damping_per_s=damping,
            dt_s=dt,
            steps=steps,
            duration_s=args.duration,
            ramp_s=args.ramp,
            min_altitude_m=altitude,
        ),
        pressure_mmhg=args.pressure_mmhg,
        pressure_Pa=args.pressure_mmhg * 133.322387415,
        device=str(device),
        device_name=device.name,
        warp_version=wp.__version__,
        precision="float32 centered coordinates",
        coordinate_origin_m=origin.tolist(),
        max_displacement_mm=history[-1][2],
        relative_free_force_residual=history[-1][3],
        max_speed_m_s=history[-1][4],
        kinetic_energy_J=history[-1][5],
        fixed_boundary_displacement_m=0.0,
        setup_wall_s=setup_seconds,
        solve_wall_s=solve_seconds,
        cuda_event_interval_s=gpu_ms / 1000.0,
        timing_note="CUDA event intervals include launch starvation; diagnostics excluded from event intervals, included in solve wall time",
        status="finite-duration damped transient; not physical validation",
    )
    from warp_febio_pilot.visualization.patch import render

    render(args.output / "state.npz", args.output / "deformation.png")
    result["total_wall_s"] = time.perf_counter() - start
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
