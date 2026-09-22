import jax
from jax import numpy as jnp

from flapjax.aero.data_structures import GridDiscretisation
from flapjax.aero.flowfields import ConstantFlowField
from flapjax.aero.utils import make_rectangular_grid, prandtl_glauert_transform
from flapjax.aero.uvlm import UVLM


class TestPrandtlGlauertTransform:
    @staticmethod
    def test_transform_geometry():
        r"""
        A point purely along the freestream direction is unchanged; a point purely perpendicular to the freestream
        is scaled by beta; beta=1 (incompressible) is the identity transform.
        """
        u_inf_dir = jnp.array((1.0, 0.0, 0.0))
        beta = jnp.array(0.8)

        r_par = jnp.array((3.0, 0.0, 0.0))
        assert jnp.allclose(prandtl_glauert_transform(r_par, u_inf_dir, beta), r_par)

        r_perp = jnp.array((0.0, 2.0, -1.0))
        assert jnp.allclose(
            prandtl_glauert_transform(r_perp, u_inf_dir, beta), beta * r_perp
        )

        r = jnp.array((1.0, 2.0, 3.0))
        assert jnp.allclose(prandtl_glauert_transform(r, u_inf_dir, jnp.array(1.0)), r)


def _build_mirrored_wing(
    mach: float | jax.Array,
    alpha_deg: float = 3.0,
    m: int = 4,
    n: int = 8,
    chord: float = 1.0,
    span: float = 8.0,
    u_mag: float = 20.0,
) -> tuple[UVLM, jax.Array]:
    r"""
    Build a small mirrored rectangular wing at a fixed geometric angle of attack, for compressibility-correction
    regression tests.
    :return: UVLM case and the lift direction (unit vector perpendicular to the freestream, in the plane containing the freestream and the global z axis).
    """
    alpha_rad = jnp.deg2rad(alpha_deg)
    u_inf = jnp.array((u_mag * jnp.cos(alpha_rad), 0.0, u_mag * jnp.sin(alpha_rad)))
    lift_dir = jnp.array((-jnp.sin(alpha_rad), 0.0, jnp.cos(alpha_rad)))
    flowfield = ConstantFlowField(
        u_inf=u_inf, rho=1.225, relative_motion=True, mach=mach
    )

    disc = GridDiscretisation(m=m, n=n, m_star=2)

    hg = jnp.zeros((n + 1, 4, 4))
    beam_coords = jnp.zeros((n + 1, 3)).at[:, 1].set(jnp.linspace(0.0, span, n + 1))
    hg = hg.at[:, :3, :3].set(jnp.eye(3)[None, :, :])
    hg = hg.at[:, :3, 3].set(beam_coords)

    x_grid = make_rectangular_grid(m=m, n=n, chord=chord, ea=0.25)
    dt = chord / (u_mag * m)

    uvlm = UVLM(
        grid_shapes=[disc],
        dof_mapping=jnp.arange(n + 1),
        mirror_point=jnp.zeros(3),
        mirror_normal=jnp.array((0.0, 1.0, 0.0)),
    )
    uvlm.set_design_variables(dt=dt, flowfield=flowfield, zeta_b0=x_grid, hg0=hg)
    return uvlm, lift_dir


def _total_lift(uvlm: UVLM, lift_dir: jax.Array) -> jax.Array:
    sol = uvlm.static_solve(horseshoe=True)
    f_tot = jnp.sum(sol.f_steady[0], axis=(0, 1))
    return jnp.dot(f_tot, lift_dir)


class TestPrandtlGlauertUvlm:
    @staticmethod
    def test_lift_increases_with_mach():
        r"""
        At fixed geometric angle of attack, lift must increase with Mach number, bounded between the incompressible
        value and the classical 2D Prandtl-Glauert prediction 1/beta (3D effects mean the finite-wing result won't
        reach 1/beta).
        """
        mach = 0.6
        beta = jnp.sqrt(1.0 - mach**2)

        uvlm_incompressible, lift_dir = _build_mirrored_wing(mach=0.0)
        uvlm_compressible, _ = _build_mirrored_wing(mach=mach)

        lift_incompressible = _total_lift(uvlm_incompressible, lift_dir)
        lift_compressible = _total_lift(uvlm_compressible, lift_dir)

        ratio = lift_compressible / lift_incompressible
        assert 1.0 < ratio < 1.0 / beta, (
            f"Lift ratio {ratio} not between 1 and the 2D Prandtl-Glauert bound {1.0 / beta}"
        )

    @staticmethod
    def test_strip_alpha_matches_geometric_under_compressibility():
        r"""
        The per-strip effective angle of attack (used for polar-correction sampling) must still match the true
        geometric angle of attack under compressibility: the sectional lift is Prandtl-Glauert corrected, so the
        alpha extraction must invert it with a 2*pi/beta lift slope rather than assuming a flat 2*pi slope. Mirrors
        test_infinite_wing_polar.py's pseudo-infinite-wing check, but at a non-zero Mach number.
        """
        alpha_deg = 3.0
        alpha_rad = jnp.deg2rad(alpha_deg)
        uvlm, _ = _build_mirrored_wing(
            mach=0.6, alpha_deg=alpha_deg, m=4, n=12, span=48.0
        )
        sol = uvlm.static_solve(horseshoe=True)

        alpha_strip = sol.alpha[0]  # (n_strip, )

        # inboard quarter of the span, excluding the mirror-plane edge strip
        n_interior = alpha_strip.shape[0] // 4
        interior = alpha_strip[1:n_interior]

        max_rel_err = jnp.max(jnp.abs(interior - alpha_rad) / alpha_rad)
        assert max_rel_err < 0.02, (
            f"Inboard strip alpha does not match geometric alpha under compressibility, "
            f"measured={interior}, expected={alpha_rad}"
        )

    @staticmethod
    def test_mach_gradient():
        r"""
        Gradient of total lift with respect to Mach number must be positive.
        """

        def _lift_of_mach(mach: jax.Array) -> jax.Array:
            uvlm, lift_dir = _build_mirrored_wing(mach=mach)
            return _total_lift(uvlm, lift_dir)

        grad_val = jax.grad(_lift_of_mach)(jnp.array(0.3))
        assert jnp.isfinite(grad_val)
        assert grad_val > 0.0
