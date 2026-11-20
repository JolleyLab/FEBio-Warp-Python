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

from . import febio_reference, model as model_api
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


def parameter_gradient(model, F, params):
    """d(sum of energy)/d(parameters) from the Warp tape. Returns a NumPy vector."""
    n = len(F)
    mat = wp.mat33d
    values = wp.array([model.pack(**params)], dtype=model.param_vector(), requires_grad=True)
    W = wp.zeros(n, dtype=wp.float64, requires_grad=True)
    sigma = wp.zeros(n, dtype=mat)
    cd = wp.zeros(n, dtype=mat)
    P = wp.zeros(n, dtype=mat)
    dP = wp.zeros(n, dtype=mat)
    eye = np.tile(np.eye(3), (n, 1, 1))
    tape = wp.Tape()
    with tape:
        wp.launch(
            model_api.make_point_kernel(model),
            dim=n,
            inputs=[
                wp.array(F, dtype=mat),
                wp.zeros(n, dtype=mat),
                wp.array(eye, dtype=mat),
                wp.zeros(n, dtype=int),
                values,
            ],
            outputs=[W, sigma, cd, P, dP],
        )
    # Seeding every energy adjoint with 1 gives the gradient of the summed energy.
    W.grad.fill_(1.0)
    tape.backward()
    return values.grad.numpy()[0].copy()


def check_model(model, params, device="cpu"):
    """Return a dict of conformance metrics for one model and one parameter set."""
    F = sample_states()
    n = len(F)
    rng = np.random.default_rng(11)
    H = 0.2 * rng.normal(size=F.shape)
    out = model_api.evaluate(model, F, params, H, device=device)
    scale = np.abs(out["cauchy"]).max()
    J = np.linalg.det(F)
    Finv = np.linalg.inv(F)
    l = H @ Finv  # velocity gradient of the perturbation H
    d = 0.5 * (l + l.transpose(0, 2, 1))
    metrics = {"states": n}

    # 1. Undeformed and rigidly rotated states carry no energy and no stress.
    R = _rotation([1.0, 2.0, -1.0], 0.9)
    rest = model_api.evaluate(model, np.array([np.eye(3), R]), params, device=device)
    metrics["rest_energy"] = float(np.abs(rest["energy"]).max() / scale)
    metrics["rest_stress"] = float(np.abs(rest["cauchy"]).max() / scale)

    # 2. Cauchy stress is symmetric (balance of angular momentum).
    metrics["cauchy_asymmetry"] = float(
        np.abs(out["cauchy"] - out["cauchy"].transpose(0, 2, 1)).max() / scale
    )

    # 3. Objectivity: W(RF) = W(F) and sigma(RF) = R sigma(F) R^T.
    rotated = model_api.evaluate(model, R @ F, params, device=device)
    metrics["objectivity_energy"] = _rel(rotated["energy"], out["energy"])
    metrics["objectivity_stress"] = _rel(rotated["cauchy"], R @ out["cauchy"] @ R.T)

    # 4. Stress is the derivative of the energy:  dW = J sigma : d.
    #    Tangent is the derivative of the stress, through the Kirchhoff stress
    #    tau = J sigma:  d(tau) = J c:d + l tau + tau l^T.
    #    Central differences along the random direction H.
    step = 1.0e-6
    plus = model_api.evaluate(model, F + step * H, params, device=device)
    minus = model_api.evaluate(model, F - step * H, params, device=device)
    metrics["stress_vs_energy_fd"] = _rel(
        J * np.einsum("nij,nij->n", out["cauchy"], d),
        (plus["energy"] - minus["energy"]) / (2 * step),
    )
    tau = J[:, None, None] * out["cauchy"]
    Jp = np.linalg.det(F + step * H)[:, None, None]
    Jm = np.linalg.det(F - step * H)[:, None, None]
    dtau_fd = (Jp * plus["cauchy"] - Jm * minus["cauchy"]) / (2 * step)
    dtau = (
        J[:, None, None] * out["spatial_tangent_action"]
        + l @ tau
        + tau @ l.transpose(0, 2, 1)
    )
    metrics["tangent_vs_stress_fd"] = _rel(dtau, dtau_fd)

    # 5. The reference-frame pair on the descriptor is the same physics:
    #    P = J sigma F^-T, and dP[H] agrees with a finite difference of P.
    metrics["pk1_vs_cauchy"] = _rel(
        out["pk1"], J[:, None, None] * out["cauchy"] @ Finv.transpose(0, 2, 1)
    )
    metrics["pk1_tangent_vs_fd"] = _rel(
        out["tangent_action"], (plus["pk1"] - minus["pk1"]) / (2 * step)
    )

    # 6. Single precision compiled out of the same source.
    single = model_api.evaluate(model, F, params, H, dtype=wp.float32, device=device)
    metrics["fp32_vs_fp64_stress"] = _rel(single["cauchy"], out["cauchy"])

    # 7. Cauchy stress and 6x6 spatial tangent against the FEBio closed forms.
    closed_form = febio_reference.CLOSED_FORMS.get(model.name)
    spatial = evaluate_exchange(model, F, params, device=device)
    metrics["tangent_asymmetry"] = float(
        np.abs(spatial["tangent"] - spatial["tangent"].transpose(0, 2, 1)).max()
        / np.abs(spatial["tangent"]).max()
    )
    if closed_form is not None:
        ref = [closed_form(Fi, **params) for Fi in F]
        metrics["cauchy_vs_febio"] = _rel(spatial["cauchy"], np.array([r[0] for r in ref]))
        metrics["tangent_vs_febio"] = _rel(spatial["tangent"], np.array([r[1] for r in ref]))

    # 8. Gradient with respect to the parameters, tape against central differences.
    if device == "cpu":
        gradient = parameter_gradient(model, F, params)
        fd = []
        for name in model.param_names:
            h = 1.0e-6 * abs(params[name])
            up = model_api.evaluate(model, F, {**params, name: params[name] + h})["energy"].sum()
            dn = model_api.evaluate(model, F, {**params, name: params[name] - h})["energy"].sum()
            fd.append((up - dn) / (2 * h))
        fd = np.array(fd)
        metrics["parameter_gradient_vs_fd"] = float(np.abs(gradient / fd - 1.0).max())
    return metrics
