"""Conformance checks that every material model must pass.

`check_model` runs one model through a fixed set of deformation states and
returns plain numbers (relative errors). The unit test asserts on them and
examples/material_interface/run.py writes them to validation/ so reviewers can
read the same figures. Nothing here is specific to one model.

Relative errors are normalized by the largest magnitude of the reference
quantity over the batch, so states near F = I do not inflate them.
"""

import numpy as np
import warp as wp

from . import model as model_api
from .exchange import evaluate_exchange


def sample_states(seed=7, count=60):
    """Named homogeneous states followed by random admissible ones. Shape (n,3,3)."""
    rng = np.random.default_rng(seed)
    named = [
        np.diag([1.30, 1.0, 1.0]),  # uniaxial stretch, lateral faces held
        np.diag([1.30, 0.88, 0.88]),  # uniaxial stretch with lateral contraction
        np.diag([1.20, 1.20, 0.70]),  # equibiaxial stretch (leaflet-like)
        np.diag([0.85, 0.85, 0.85]),  # volumetric compression
        np.diag([1.10, 1.10, 1.10]),  # volumetric expansion
        np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),  # simple shear
    ]
    random = np.eye(3) + 0.15 * rng.normal(size=(4 * count, 3, 3))
    random = random[np.linalg.det(random) > 0.4][:count]
    return np.concatenate([np.array(named), random])


def _rel(value, reference):
    return float(np.abs(value - reference).max() / np.abs(reference).max())


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def parameter_gradient(model, F, params, geometry=None):
    """d(sum of energy)/d(parameters) from the Warp tape. Returns a NumPy vector."""
    n = len(F)
    mat = wp.mat33d
    values = wp.array([model.pack(**params)], dtype=model.param_vector(), requires_grad=True)
    W = wp.zeros(n, dtype=wp.float64, requires_grad=True)
    sigma = wp.zeros(n, dtype=mat)
    cd = wp.zeros(n, dtype=mat)
    P = wp.zeros(n, dtype=mat)
    dP = wp.zeros(n, dtype=mat)
    geometry = model.pack_geometry(n, **(geometry or {}))
    tape = wp.Tape()
    with tape:
        wp.launch(
            model_api.make_point_kernel(model),
            dim=n,
            inputs=[
                wp.array(F, dtype=mat),
                wp.zeros(n, dtype=mat),
                wp.array(geometry, dtype=model.geometry_vector()),
                wp.zeros(n, dtype=int),
                values,
            ],
            outputs=[W, sigma, cd, P, dP, wp.zeros(n, dtype=int)],
        )
    # Seeding every energy adjoint with 1 gives the gradient of the summed energy.
    W.grad.fill_(1.0)
    tape.backward()
    return values.grad.numpy()[0].copy()


def _energy_and_geometry_gradient(model, F, params, g, tape):
    n = len(F)
    g_wp = wp.array(g, dtype=model.geometry_vector(), requires_grad=tape)
    W = wp.zeros(n, dtype=wp.float64, requires_grad=tape)
    inputs = [
        wp.array(F, dtype=wp.mat33d),
        wp.zeros(n, dtype=wp.mat33d),
        g_wp,
        wp.zeros(n, dtype=int),
        wp.array([model.pack(**params)], dtype=model.param_vector()),
    ]
    outputs = [W] + [wp.zeros(n, dtype=wp.mat33d) for _ in range(4)]
    outputs.append(wp.zeros(n, dtype=int))
    kernel = model_api.make_point_kernel(model)
    if not tape:
        wp.launch(kernel, dim=n, inputs=inputs, outputs=outputs)
        return W.numpy().sum()
    recording = wp.Tape()
    with recording:
        wp.launch(kernel, dim=n, inputs=inputs, outputs=outputs)
    W.grad.fill_(1.0)
    recording.backward()
    return g_wp.grad.numpy().sum(axis=0)


def stress_gradient_errors(model, F, params, geometry=None, seed=23):
    """Reverse-mode derivatives of the Cauchy stress against central differences.

    Inverse problems and implicit differentiation at equilibrium differentiate
    the stress (through the residual), not the energy. With a fixed random
    weight W per point, the scalar L = sum over points of W : sigma is
    differentiated with respect to the parameters and with respect to F.
    Returns (parameter error, deformation gradient error), each relative to the
    largest entry of the finite-difference gradient.
    """
    n = len(F)
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(n, 3, 3))
    g = model.pack_geometry(n, **(geometry or {}))
    kernel = model_api.make_point_kernel(model)

    def launch(F_values, values, tape):
        F_wp = wp.array(F_values, dtype=wp.mat33d, requires_grad=tape)
        p_wp = wp.array([model.pack(**values)], dtype=model.param_vector(), requires_grad=tape)
        sigma = wp.zeros(n, dtype=wp.mat33d, requires_grad=tape)
        outputs = [
            wp.zeros(n, dtype=wp.float64),
            sigma,
            wp.zeros(n, dtype=wp.mat33d),
            wp.zeros(n, dtype=wp.mat33d),
            wp.zeros(n, dtype=wp.mat33d),
            wp.zeros(n, dtype=int),
        ]
        inputs = [
            F_wp,
            wp.zeros(n, dtype=wp.mat33d),
            wp.array(g, dtype=model.geometry_vector()),
            wp.zeros(n, dtype=int),
            p_wp,
        ]
        if not tape:
            wp.launch(kernel, dim=n, inputs=inputs, outputs=outputs)
            return float((sigma.numpy() * weights).sum())
        recording = wp.Tape()
        with recording:
            wp.launch(kernel, dim=n, inputs=inputs, outputs=outputs)
        # Seeding the stress adjoint with the weights gives dL/d(inputs).
        sigma.grad.assign(wp.array(weights, dtype=wp.mat33d))
        recording.backward()
        return p_wp.grad.numpy()[0].copy(), F_wp.grad.numpy().copy()

    dL_dp, dL_dF = launch(F, params, True)

    numeric_p, tape_p = [], []
    for index, name in enumerate(model.param_names):
        h = 1.0e-6 * abs(params[name]) if params[name] != 0.0 else 1.0e-9
        ranges = [r for r in model.param_ranges if r.name == name]
        if any(not (r.check(params[name] + h) and r.check(params[name] - h)) for r in ranges):
            continue
        up = launch(F, {**params, name: params[name] + h}, False)
        down = launch(F, {**params, name: params[name] - h}, False)
        numeric_p.append((up - down) / (2 * h))
        tape_p.append(dL_dp[index])
    numeric_p, tape_p = np.array(numeric_p), np.array(tape_p)
    error_p = float(np.abs(tape_p - numeric_p).max() / max(np.abs(numeric_p).max(), 1e-300)) if len(numeric_p) else 0.0

    # dL/dF: perturb one entry of F at every point at once. Points are
    # independent, so the change in L is the sum of the per-point derivatives
    # for that entry.
    numeric_F = np.zeros(9)
    h = 1.0e-6
    for k in range(9):
        step = np.zeros((3, 3))
        step.flat[k] = h
        numeric_F[k] = (launch(F + step, params, False) - launch(F - step, params, False)) / (2 * h)
    tape_F = dL_dF.sum(axis=0).reshape(9)
    error_F = float(np.abs(tape_F - numeric_F).max() / max(np.abs(numeric_F).max(), 1e-300))
    return error_p, error_F


def geometry_gradient_error(model, F, params, geometry=None):
    """Tape gradient of summed energy with respect to every geometry entry,
    against central differences. Returns the largest error relative to the
    largest gradient entry."""
    g = model.pack_geometry(len(F), **(geometry or {}))
    from_tape = _energy_and_geometry_gradient(model, F, params, g, True)
    h = 1.0e-6
    numeric = np.zeros(g.shape[1])
    for k in range(g.shape[1]):
        step = np.zeros(g.shape[1])
        step[k] = h
        numeric[k] = (
            _energy_and_geometry_gradient(model, F, params, g + step, False)
            - _energy_and_geometry_gradient(model, F, params, g - step, False)
        ) / (2 * h)
    scale = max(np.abs(numeric).max(), 1.0e-300)
    return float(np.abs(from_tape - numeric).max() / scale)


def check_model(model, params, closed_form=None, geometry=None, states=None, device="cpu"):
    """Return a dict of conformance metrics for one model and one parameter set.

    geometry: optional dict of named geometry fields, each of shape (size,),
    applied at every point.
    states: optional deformation gradients (n,3,3) to use instead of
    sample_states(), for parameter sets that are only meaningful at small strain.
    closed_form: optional function F -> (Cauchy 3x3, tangent 6x6) transcribed
    independently from the FEBio source. These live with the tests, not in the
    package.
    """
    F = sample_states() if states is None else np.asarray(states, dtype=np.float64)
    n = len(F)
    rng = np.random.default_rng(11)
    H = 0.2 * rng.normal(size=F.shape)
    out = model_api.evaluate(model, F, params, H, geometry=geometry, device=device)
    scale = np.abs(out["cauchy"]).max()
    J = np.linalg.det(F)
    Finv = np.linalg.inv(F)
    l = H @ Finv  # velocity gradient of the perturbation H
    d = 0.5 * (l + l.transpose(0, 2, 1))
    metrics = {"states": n}

    # 1. Undeformed and rigidly rotated states carry no energy and no stress.
    R = _rotation([1.0, 2.0, -1.0], 0.9)
    rest = model_api.evaluate(model, np.array([np.eye(3), R]), params, geometry=geometry, device=device)
    metrics["rest_energy"] = float(np.abs(rest["energy"]).max() / scale)
    metrics["rest_stress"] = float(np.abs(rest["cauchy"]).max() / scale)

    # 2. Cauchy stress is symmetric (balance of angular momentum).
    metrics["cauchy_asymmetry"] = float(np.abs(out["cauchy"] - out["cauchy"].transpose(0, 2, 1)).max() / scale)

    # 3. Objectivity: W(RF) = W(F) and sigma(RF) = R sigma(F) R^T.
    rotated = model_api.evaluate(model, R @ F, params, geometry=geometry, device=device)
    metrics["objectivity_energy"] = _rel(rotated["energy"], out["energy"])
    metrics["objectivity_stress"] = _rel(rotated["cauchy"], R @ out["cauchy"] @ R.T)

    # 4. Stress is the derivative of the energy:  dW = J sigma : d.
    #    Tangent is the derivative of the stress, through the Kirchhoff stress
    #    tau = J sigma:  d(tau) = J c:d + l tau + tau l^T.
    #    Central differences along the random direction H.
    step = 1.0e-6
    plus = model_api.evaluate(model, F + step * H, params, geometry=geometry, device=device)
    minus = model_api.evaluate(model, F - step * H, params, geometry=geometry, device=device)
    metrics["stress_vs_energy_fd"] = _rel(
        J * np.einsum("nij,nij->n", out["cauchy"], d),
        (plus["energy"] - minus["energy"]) / (2 * step),
    )
    tau = J[:, None, None] * out["cauchy"]
    Jp = np.linalg.det(F + step * H)[:, None, None]
    Jm = np.linalg.det(F - step * H)[:, None, None]
    dtau_fd = (Jp * plus["cauchy"] - Jm * minus["cauchy"]) / (2 * step)
    dtau = J[:, None, None] * out["spatial_tangent_action"] + l @ tau + tau @ l.transpose(0, 2, 1)
    metrics["tangent_vs_stress_fd"] = _rel(dtau, dtau_fd)

    # 5. The reference-frame pair on the descriptor is the same physics:
    #    P = J sigma F^-T, and dP[H] agrees with a finite difference of P.
    metrics["pk1_vs_cauchy"] = _rel(out["pk1"], J[:, None, None] * out["cauchy"] @ Finv.transpose(0, 2, 1))
    metrics["pk1_tangent_vs_fd"] = _rel(out["tangent_action"], (plus["pk1"] - minus["pk1"]) / (2 * step))

    # 6. Single precision compiled out of the same source.
    single = model_api.evaluate(model, F, params, H, dtype=wp.float32, geometry=geometry, device=device)
    metrics["fp32_vs_fp64_stress"] = _rel(single["cauchy"], out["cauchy"])

    # 7. Cauchy stress and 6x6 spatial tangent against the FEBio closed forms,
    #    when the caller supplies them.
    spatial = evaluate_exchange(model, F, params, geometry=geometry, device=device)
    metrics["tangent_asymmetry"] = float(
        np.abs(spatial["tangent"] - spatial["tangent"].transpose(0, 2, 1)).max() / np.abs(spatial["tangent"]).max()
    )
    if model.stress_and_tangent is not None:
        # The one-pass stress and tangent must equal stress plus six actions.
        # `spatial` above used the one-pass path; compare with the other.
        actions = evaluate_exchange(model, F, params, geometry=geometry, device=device, combined=False)
        metrics["combined_vs_actions"] = max(
            _rel(spatial["tangent"], actions["tangent"]),
            _rel(spatial["cauchy"], actions["cauchy"]),
        )
    if closed_form is not None:
        ref = [closed_form(Fi, **params) for Fi in F]
        metrics["cauchy_vs_febio"] = _rel(spatial["cauchy"], np.array([r[0] for r in ref]))
        metrics["tangent_vs_febio"] = _rel(spatial["tangent"], np.array([r[1] for r in ref]))

    # 8. Gradient with respect to the parameters, tape against central differences.
    with wp.ScopedDevice(device):
        gradient = parameter_gradient(model, F, params, geometry)
        # A parameter sitting on a bound of its admissible range cannot be
        # perturbed to both sides, so it is left out of the comparison.
        on_bound = {r.name for r in model.param_ranges if r.name in params and params[r.name] in (r.low, r.high)}
        errors = []
        for index, name in enumerate(model.param_names):
            if name in on_bound:
                continue
            # Relative step, so that very small parameter values (the FEBio suite
            # uses moduli down to 1e-12) are perturbed in proportion.
            h = 1.0e-6 * abs(params[name]) if params[name] != 0.0 else 1.0e-9
            ranges = [r for r in model.param_ranges if r.name == name]
            if any(not (r.check(params[name] + h) and r.check(params[name] - h)) for r in ranges):
                continue
            up = model_api.evaluate(model, F, {**params, name: params[name] + h}, geometry=geometry)["energy"].sum()
            dn = model_api.evaluate(model, F, {**params, name: params[name] - h}, geometry=geometry)["energy"].sum()
            numeric = (up - dn) / (2 * h)
            errors.append(abs(gradient[index] - numeric))
        scale = np.abs(gradient).max()
        metrics["parameter_gradient_vs_fd"] = float(max(errors) / scale) if errors else 0.0

        # 9. Reverse-mode derivatives of the stress itself, with respect to the
        #    parameters and to F.
        error_p, error_F = stress_gradient_errors(model, F, params, geometry)
        metrics["stress_parameter_gradient_vs_fd"] = error_p
        metrics["stress_deformation_gradient_vs_fd"] = error_F

        # 10. Gradient with respect to the geometry vector (material axes, fiber
        #    directions), for models that use one. Entries are perturbed
        #    independently; no orthonormality is imposed here.
        if model.geometry:
            metrics["geometry_gradient_vs_fd"] = geometry_gradient_error(model, F, params, geometry)
    return metrics
