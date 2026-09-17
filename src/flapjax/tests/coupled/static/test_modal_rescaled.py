from __future__ import annotations

import jax.numpy as jnp

from flapjax.models.cantilever_wing.cantilever_wing import generate_cantilever_wing

C_REF = 1.0
RHO_REF = 1.225
U_REF = 10.0
PRESCRIBED_DOFS = tuple(range(6))

U_TEST = 20.0
RHO_TEST = 0.8


def _build_and_linearise(u_inf_mag: float, rho: float):
    wing = generate_cantilever_wing(
        n_nodes=8,
        c_ref=1.0,
        m=4,
        m_star=12,
        u_inf=jnp.array((u_inf_mag, 0.0, 0.0)),
        rho=rho,
    )
    ref = wing.static_solve(prescribed_dofs=PRESCRIBED_DOFS, horseshoe=False)
    return wing.linearise(
        reference=ref,
        skip_checks=True,
        n_struct_modes=None,
    )


class TestModalRescaled:
    @classmethod
    def setup_class(cls):
        cls.linear_ref = _build_and_linearise(U_REF, RHO_REF)
        cls.linear_test = _build_and_linearise(U_TEST, RHO_TEST)

    def test_identity_rescale(self):
        """Rescaling with the reference inputs should give the same result."""
        n_modes = 8
        kwargs = {
            "n_modes": n_modes,
            "remove_complex_conjugate": True,
            "sort": "frequency",
            "min_struct_content": 0.0,
        }
        evals_modal = self.linear_ref.modal(**kwargs)
        evals_rescaled = self.linear_ref.modal_rescaled(
            velocity=U_REF, density=RHO_REF, chord=C_REF, **kwargs
        )

        assert jnp.allclose(evals_modal, evals_rescaled, rtol=1e-10, atol=1e-12), (
            f"Identity rescale differs from modal():\n"
            f"modal:    {evals_modal}\n"
            f"rescaled: {evals_rescaled}"
        )

    def test_velocity_density_rescale(self):
        """Rescaled aeroelastic eigenvalues must match a full re-linearisation."""
        n_modes = 8
        kwargs = {
            "n_modes": n_modes,
            "remove_complex_conjugate": True,
            "sort": "frequency",
            "min_struct_content": 0.05,
        }
        evals_rescaled = self.linear_ref.modal_rescaled(
            velocity=U_TEST, density=RHO_TEST, chord=C_REF, **kwargs
        )
        evals_truth = self.linear_test.modal(**kwargs)

        # pair by damped frequency then compare both real and imaginary parts
        idx_r = jnp.argsort(jnp.abs(evals_rescaled.imag))
        idx_t = jnp.argsort(jnp.abs(evals_truth.imag))

        assert jnp.allclose(
            evals_rescaled[idx_r].imag,
            evals_truth[idx_t].imag,
            rtol=5e-3,
        ), (
            f"Damped frequencies differ:\n"
            f"rescaled: {evals_rescaled[idx_r].imag}\n"
            f"truth:    {evals_truth[idx_t].imag}"
        )

        assert jnp.allclose(
            evals_rescaled[idx_r].real,
            evals_truth[idx_t].real,
            rtol=5e-3,
        ), (
            f"Damping (real parts) differ:\n"
            f"rescaled: {evals_rescaled[idx_r].real}\n"
            f"truth:    {evals_truth[idx_t].real}"
        )
