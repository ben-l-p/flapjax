from typing import Any

from jax import Array
from jax import numpy as jnp

from flapjax.aero.data_structures import GridDiscretisation
from flapjax.aero.flowfields import ConstantFlowField
from flapjax.aero.utils import make_rectangular_grid
from flapjax.aero.uvlm import UVLM, PolarFunction


def _build_pseudo_infinite_wing(
    alpha_deg: float,
    polar_function: PolarFunction | None,
    polar_data: Any,
    m: int = 4,
    n: int = 12,
    chord: float = 1.0,
    span: float = 48.0,
    u_mag: float = 20.0,
    ea: float = 0.25,
    polar_circulation_scale: float = 0.0,
) -> tuple[UVLM, Array, float, float, float]:
    r"""
    Build a pseudo-infinite mirrored wing at a fixed geometric angle of attack. This behaves like a 2D section
    for the inboard strips.
    """
    alpha_rad = jnp.deg2rad(alpha_deg)
    u_inf = jnp.array((u_mag * jnp.cos(alpha_rad), 0.0, u_mag * jnp.sin(alpha_rad)))
    flowfield = ConstantFlowField(u_inf=u_inf, rho=1.225, relative_motion=True)

    disc = GridDiscretisation(m=m, n=n, m_star=2)

    hg = jnp.zeros((n + 1, 4, 4))
    beam_coords = jnp.zeros((n + 1, 3)).at[:, 1].set(jnp.linspace(0.0, span, n + 1))
    hg = hg.at[:, :3, :3].set(jnp.eye(3)[None, :, :])
    hg = hg.at[:, :3, 3].set(beam_coords)

    x_grid = make_rectangular_grid(m=m, n=n, chord=chord, ea=ea)
    dt = chord / (u_mag * m)

    uvlm = UVLM(
        grid_shapes=[disc],
        dof_mapping=jnp.arange(n + 1),
        mirror_point=jnp.zeros(3),
        mirror_normal=jnp.array((0.0, 1.0, 0.0)),
        polar_data=[polar_data],
        polar_function=[polar_function],
        polar_circulation_scale=polar_circulation_scale,
    )
    uvlm.set_design_variables(dt=dt, flowfield=flowfield, zeta_b0=x_grid, hg0=hg)
    return uvlm, alpha_rad, u_mag, chord, span


class TestPseudoInfiniteWing:
    @staticmethod
    def test_strip_aoa_matches_geometric():
        r"""
        Ensure that the measured AoA from the output matches the geometric AoA for the inboard strips.
        """
        alpha_deg = 3.0
        uvlm, alpha_rad, *_ = _build_pseudo_infinite_wing(
            alpha_deg=alpha_deg, polar_function=None, polar_data=None
        )
        sol = uvlm.static_solve(horseshoe=True)

        alpha_strip = sol.alpha[0]  # (n_strip, )

        # inboard quarter of the span, excluding the mirror-plane edge strip
        n_interior = alpha_strip.shape[0] // 4
        interior = alpha_strip[1:n_interior]

        max_rel_err = jnp.max(jnp.abs(interior - alpha_rad) / alpha_rad)
        assert max_rel_err < 0.02, (
            f"Inboard strip alpha does not match from geometric alpha, measued={interior}, expected={alpha_rad}"
        )

    @staticmethod
    def test_polar_correction_lift():
        r"""
        Test a polar with constant lift slope and zero drag/moment.
        """
        alpha_deg = 3.0

        def polar_half(a: Array, data: Any) -> tuple[Array, Array, Array]:
            # lift only
            return data * a, jnp.zeros_like(a), jnp.zeros_like(a)

        uvlm, alpha_rad, _, chord, span = _build_pseudo_infinite_wing(
            alpha_deg=alpha_deg, polar_function=None, polar_data=None
        )
        n_strip = uvlm.grid_disc[0].n
        uvlm, *_ = _build_pseudo_infinite_wing(
            alpha_deg=alpha_deg, polar_function=polar_half, polar_data=jnp.pi
        )
        sol = uvlm.static_solve(horseshoe=True)

        measured_force = jnp.sum(
            sol.f_steady[0] * jnp.array((-jnp.sin(alpha_rad), 0.0, jnp.cos(alpha_rad)))
        )  # rotation into freestream frame and sum force over surface

        b_strip = span / n_strip  # span of each strip
        lift_expected = (  # use polars from aoa extracted from model
            sol.flowfield.q_inf * chord * b_strip * jnp.pi * jnp.sum(sol.alpha[0])
        )

        rel_err = jnp.abs(measured_force - lift_expected) / jnp.abs(lift_expected)
        assert rel_err < 1e-6, (
            f"Mismatch in forces, measured: {float(measured_force):.6e}, expected: {float(lift_expected):.6e}"
        )

    @staticmethod
    def test_polar_correction_circulation_scaling():
        r"""
        With a reduced lift slope and  ``polar_circulation_scale=1.0``, the bound and
        wake circulations should be linearly scaled relative to the base UVLM solution.
        """
        alpha_deg = 3.0

        def polar_half(a: Array, data: Array) -> tuple[Array, Array, Array]:
            return data * a, jnp.zeros_like(a), jnp.zeros_like(a)

        uvlm_ref, *_ = _build_pseudo_infinite_wing(
            alpha_deg=alpha_deg, polar_function=None, polar_data=None
        )

        sol_ref = uvlm_ref.static_solve(horseshoe=True)

        uvlm_corr, *_ = _build_pseudo_infinite_wing(
            alpha_deg=alpha_deg,
            polar_function=polar_half,
            polar_data=jnp.pi,
            polar_circulation_scale=1.0,
        )
        sol_corr = uvlm_corr.static_solve(horseshoe=True)

        # scale should be half
        gamma_b_ratio = sol_corr.gamma_b[0] / sol_ref.gamma_b[0]
        gamma_w_ratio = sol_corr.gamma_w[0] / sol_ref.gamma_w[0]

        assert jnp.max(jnp.abs(gamma_b_ratio - 0.5)) < 1e-6, (
            f"Bound circulation not scaled to 0.5, max deviation "
            f"{float(jnp.max(jnp.abs(gamma_b_ratio - 0.5))):.6e}"
        )
        assert jnp.max(jnp.abs(gamma_w_ratio - 0.5)) < 1e-6, (
            f"Wake circulation not scaled to 0.5, max deviation "
            f"{float(jnp.max(jnp.abs(gamma_w_ratio - 0.5))):.6e}"
        )

    @staticmethod
    def test_polar_correction_drag():
        r"""
        Test a wing with a constant drag coefficient (independent of alpha).
        """
        alpha_deg = 3.0
        cd0 = 0.02

        def polar_drag_only(a: Array, data: Array) -> tuple[Array, Array, Array]:
            return jnp.zeros_like(a), jnp.full_like(a, data), jnp.zeros_like(a)

        uvlm, alpha_rad, _, chord, span = _build_pseudo_infinite_wing(
            alpha_deg=alpha_deg,
            polar_function=None,
            polar_data=None,
        )
        n_strip = uvlm.grid_disc[0].n

        uvlm, *_ = _build_pseudo_infinite_wing(
            alpha_deg=alpha_deg, polar_function=polar_drag_only, polar_data=cd0
        )
        sol = uvlm.static_solve(horseshoe=True)

        e_drag = jnp.array((jnp.cos(alpha_rad), 0.0, jnp.sin(alpha_rad)))
        drag_polar = jnp.sum(sol.f_steady[0] * e_drag)

        drag_measured = sol.flowfield.q_inf * chord * (span / n_strip) * cd0 * n_strip

        rel_err = jnp.abs(drag_polar - drag_measured) / jnp.abs(drag_measured)
        assert rel_err < 1e-6, (
            f"Expected drag {float(drag_polar):.06e} does not match measured drag {float(drag_measured):.06e}"
        )
