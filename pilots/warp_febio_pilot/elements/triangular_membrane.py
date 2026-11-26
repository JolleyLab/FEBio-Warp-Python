"""StVK plane-stress membrane; SI units, float32 Warp kernels."""

import warp as wp


@wp.kernel
def assemble(
    x: wp.array(dtype=wp.vec3),
    faces: wp.array(dtype=wp.vec3i),
    inv: wp.array(dtype=wp.mat22),
    area: wp.array(dtype=float),
    force: wp.array(dtype=wp.vec3),
    pressure_force: wp.array(dtype=wp.vec3),
    thickness: float,
    lame: float,
    mu: float,
    pressure: wp.array(dtype=float),
):
    k = wp.tid()
    ids = faces[k]
    a = x[ids[1]] - x[ids[0]]
    b = x[ids[2]] - x[ids[0]]
    # u and v below are columns of the surface deformation gradient.
    d = inv[k]
    u = a * d[0, 0] + b * d[1, 0]
    v = a * d[0, 1] + b * d[1, 1]
    # Green strain E=(F.T F-I)/2; differentiate StVK energy to get forces.
    e00 = 0.5 * (wp.dot(u, u) - 1.0)
    e11 = 0.5 * (wp.dot(v, v) - 1.0)
    e01 = 0.5 * wp.dot(u, v)
    s00 = lame * (e00 + e11) + 2.0 * mu * e00
    s11 = lame * (e00 + e11) + 2.0 * mu * e11
    s01 = 2.0 * mu * e01
    p0 = u * s00 + v * s01
    p1 = u * s01 + v * s11
    # Energy gradient, mapped back through the triangle reference basis.
    f1 = -area[k] * thickness * (p0 * d[0, 0] + p1 * d[0, 1])
    f2 = -area[k] * thickness * (p0 * d[1, 0] + p1 * d[1, 1])
    # Supplied normals face the cavity; pressure pushes away from the cavity.
    fp = -pressure[0] * wp.cross(a, b) / 6.0
    wp.atomic_add(force, ids[0], -f1 - f2 + fp)
    wp.atomic_add(force, ids[1], f1 + fp)
    wp.atomic_add(force, ids[2], f2 + fp)
    for j in range(3):
        wp.atomic_add(pressure_force, ids[j], fp)


@wp.kernel
def advance(
    x: wp.array(dtype=wp.vec3),
    rest: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    fixed: wp.array(dtype=int),
    dt: float,
    damping: float,
):
    i = wp.tid()
    if fixed[i] != 0:
        x[i] = rest[i]
        velocity[i] = wp.vec3(0.0)
    else:
        v = (velocity[i] + dt * force[i] / mass[i]) / (1.0 + dt * damping)
        velocity[i] = v
        x[i] = x[i] + dt * v


@wp.kernel
def ramp_pressure(
    counter: wp.array(dtype=int),
    pressure: wp.array(dtype=float),
    dt: float,
    ramp: float,
    target: float,
):
    counter[0] = counter[0] + 1
    fraction = wp.min(1.0, float(counter[0]) * dt / ramp)
    pressure[0] = target * float(fraction * fraction * (3.0 - 2.0 * fraction))
