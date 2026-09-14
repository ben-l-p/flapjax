from __future__ import annotations

from pathlib import Path
from typing import cast

from jax import Array, vmap
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation as Rot

from flapjax.aero.data_structures import AeroCase, GridDiscretisation
from flapjax.aero.flowfields import (
    ConstantFlowField,
    FlowField,
    OneMinusCosineFlowField,
)
from flapjax.aero.linear.data_structures import AeroInputUnflattened
from flapjax.aero.utils import biot_savart_cutoff, make_rectangular_grid
from flapjax.aero.uvlm import UVLM
from flapjax.algebra.array_utils import ArrayList


class TestLinearAero:
    u_inf: Array = jnp.array((10.0, 0.0, 1.0))
    rho_inf: float = 1.225
    m = 4
    n = 8
    m_star = 10
    physical_time = 1.0  # seconds

    c_ref = 1.0
    b_ref = 5.0
    alpha = jnp.deg2rad(0.0)

    dt = c_ref / (m * jnp.linalg.norm(u_inf))
    n_tstep = int(jnp.ceil(physical_time / dt))
    disc = GridDiscretisation(m, n, m_star)

    @classmethod
    def make_planar_wing(
        cls,
        flowfield: FlowField,
        ea: float = 0.0,
    ) -> tuple[UVLM, AeroCase, Array]:
        r"""
        Returns a reference wing case, and the reference beam coordinates.
        """

        x_grid = make_rectangular_grid(cls.m, cls.n, cls.c_ref, ea)

        beam_coords = jnp.zeros((cls.n + 1, 3))
        beam_coords = beam_coords.at[:, 1].set(jnp.linspace(0.0, cls.b_ref, cls.n + 1))
        rmat = Rot.from_euler("xyz", jnp.array((0.0, cls.alpha, 0.0))).as_matrix()

        # static position
        hg = jnp.zeros((cls.n + 1, 4, 4))
        hg = hg.at[:, 3, 3].set(1.0)
        hg = hg.at[:, :3, :3].set(rmat[None, :, :])
        hg = hg.at[:, :3, 3].set(beam_coords)

        # nonlinear case
        uvlm = UVLM(
            [cls.disc],
            jnp.arange(0, cls.n + 1),
            kernel=biot_savart_cutoff,
            gamma_dot_relaxation=1.0,
        )
        uvlm.set_design_variables(cls.dt, flowfield, x_grid, hg)
        case = uvlm.static_solve()

        return uvlm, case, hg

    @classmethod
    def test_linear_heaving_wing(cls, plot: bool = False, use_matrix: bool = False):
        flowfield = ConstantFlowField(
            TestLinearAero.u_inf, TestLinearAero.rho_inf, True
        )
        uvlm, static_case, hg0 = TestLinearAero.make_planar_wing(flowfield)

        # heaving motion
        freq = 3.0  # Hz
        omega = 0.5 * jnp.pi * freq
        ampl = 0.1  # m
        t = jnp.arange(cls.n_tstep) * cls.dt
        z_t = ampl * 0.5 * (1.0 - jnp.cos(omega * t))
        z_dot_t = ampl * omega * 0.5 * jnp.sin(omega * t)

        hg_t = jnp.zeros((cls.n_tstep, cls.n + 1, 4, 4))
        hg_t = hg_t.at[:, 3, 3].set(1.0)
        hg_t = hg_t.at[...].set(hg0[None, ...])
        hg_t = hg_t.at[:, :, 2, 3].add(z_t[:, None])

        hg_dot_t = jnp.zeros_like(hg_t)
        hg_dot_t = hg_dot_t.at[:, :, 2, 3].set(z_dot_t[:, None])

        # nonlinear case
        dynamic_case = uvlm.prescribed_dynamic_solve(
            init_case=static_case, hg_t=hg_t, hg_dot_t=hg_dot_t
        )
        if plot:
            dynamic_case.plot(Path("./test_outputs/heaving_test_nonlinear"))

        # linear case
        linear_model = uvlm.linearise(
            reference=static_case,
            wake_type="prescribed",
            bound_upwash=False,
            wake_upwash=False,
            unsteady_force=True,
        )

        delta_zeta_b = dynamic_case.zeta_b - ArrayList(
            [zeta[None, ...] for zeta in linear_model.reference.zeta_b]
        )
        u_linear = AeroInputUnflattened(
            zeta_b=delta_zeta_b,
            zeta_b_dot=dynamic_case.zeta_b_dot,
            nu_b=None,
            nu_w=None,
        )

        # Run linear case with and without matrix form
        linear_case = linear_model.run(u=u_linear)

        if plot:
            linear_case.plot(
                Path(
                    f"./test_outputs/heaving_test_linear_{'matrix' if use_matrix else 'operator'}"
                )
            )

        assert jnp.allclose(
            dynamic_case.gamma_b[0], linear_case.x_t_tot.gamma_b[0], atol=5e-3
        ), "Bound circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.gamma_w[0], linear_case.x_t_tot.gamma_w[0], atol=5e-3
        ), "Wake circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.zeta_w[0],
            cast(ArrayList, linear_case.x_t_tot.zeta_w)[0],
            atol=1e-3,
        ), "Wake grid coordinates do not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.f_steady[0], linear_case.y_t_tot.f_steady[0], atol=1e-1
        ), "Steady forces do not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.f_unsteady[0],
            cast(ArrayList, linear_case.y_t_tot.f_unsteady)[0],
            atol=2e-2,
        ), "Unsteady forces do not match between nonlinear and linear cases."

    @classmethod
    def test_linear_pitching_wing(cls, plot: bool = False, use_matrix: bool = False):
        flowfield = ConstantFlowField(
            TestLinearAero.u_inf, TestLinearAero.rho_inf, True
        )
        uvlm, static_case, hg0 = TestLinearAero.make_planar_wing(flowfield)

        # heaving motion
        freq = 3.0  # Hz
        omega = 0.5 * jnp.pi * freq
        ampl = jnp.deg2rad(4.0)  # deg
        t = jnp.arange(cls.n_tstep) * cls.dt
        alpha_t = ampl * 0.5 * (1.0 - jnp.cos(omega * t))
        alpha_dot_t = ampl * omega * 0.5 * jnp.sin(omega * t)

        rmat_t = vmap(
            lambda angle: Rot.from_euler(
                "xyz", jnp.array((0.0, angle, 0.0))
            ).as_matrix()
        )(alpha_t)

        hg_t = jnp.zeros((cls.n_tstep, cls.n + 1, 4, 4))
        hg_t = hg_t.at[...].set(hg0[None, ...])
        hg_t = hg_t.at[:, :, :3, :3].set(rmat_t[:, None, :, :])

        hg_dot_t = jnp.zeros_like(hg_t)
        hg_dot_t = hg_dot_t.at[:, :, 0, 2].set(alpha_dot_t[:, None])
        hg_dot_t = hg_dot_t.at[:, :, 2, 0].set(-alpha_dot_t[:, None])

        # nonlinear case
        dynamic_case = uvlm.prescribed_dynamic_solve(
            init_case=static_case, hg_t=hg_t, hg_dot_t=hg_dot_t
        )
        if plot:
            dynamic_case.plot(Path("./test_outputs/pitching_test_nonlinear"))

        # linear case
        linear_model = uvlm.linearise(
            reference=static_case,
            wake_type="prescribed",
            bound_upwash=False,
            wake_upwash=False,
            unsteady_force=True,
        )

        delta_zeta_b = dynamic_case.zeta_b - ArrayList(
            [zeta[None, ...] for zeta in linear_model.reference.zeta_b]
        )
        u_linear = AeroInputUnflattened(
            zeta_b=delta_zeta_b,
            zeta_b_dot=dynamic_case.zeta_b_dot,
            nu_b=None,
            nu_w=None,
        )

        # Run linear case with and without matrix form
        linear_case = linear_model.run(u_linear)

        if plot:
            linear_case.plot(
                Path(
                    f"./test_outputs/pitching_test_linear_{'matrix' if use_matrix else 'operator'}"
                )
            )

        assert jnp.allclose(
            dynamic_case.gamma_b[0], linear_case.x_t_tot.gamma_b[0], atol=6e-2
        ), "Bound circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.gamma_w[0], linear_case.x_t_tot.gamma_w[0], atol=6e-2
        ), "Wake circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.zeta_w[0],
            cast(ArrayList, linear_case.x_t_tot.zeta_w)[0],
            atol=1e-3,
        ), "Wake grid coordinates do not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.f_steady[0], linear_case.y_t_tot.f_steady[0], atol=3e-1
        ), "Steady forces do not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.f_unsteady[0],
            cast(ArrayList, linear_case.y_t_tot.f_unsteady)[0],
            atol=1e-1,
        ), "Unsteady forces do not match between nonlinear and linear cases."

    @classmethod
    def test_linear_pitching_wing_frozen_wake(
        cls, plot: bool = False, use_matrix: bool = False
    ):
        flowfield = ConstantFlowField(
            TestLinearAero.u_inf, TestLinearAero.rho_inf, True
        )
        uvlm, static_case, hg0 = TestLinearAero.make_planar_wing(flowfield, ea=1.0)

        # heaving motion
        freq = 3.0  # Hz
        omega = 0.5 * jnp.pi * freq
        ampl = jnp.deg2rad(4.0)  # deg
        t = jnp.arange(cls.n_tstep) * cls.dt
        alpha_t = ampl * 0.5 * (1.0 - jnp.cos(omega * t))
        alpha_dot_t = ampl * omega * 0.5 * jnp.sin(omega * t)

        rmat_t = vmap(
            lambda angle: Rot.from_euler(
                "xyz", jnp.array((0.0, angle, 0.0))
            ).as_matrix()
        )(alpha_t)

        hg_t = jnp.zeros((cls.n_tstep, cls.n + 1, 4, 4))
        hg_t = hg_t.at[...].set(hg0[None, ...])
        hg_t = hg_t.at[:, :, :3, :3].set(rmat_t[:, None, :, :])

        hg_dot_t = jnp.zeros_like(hg_t)
        hg_dot_t = hg_dot_t.at[:, :, 0, 2].set(alpha_dot_t[:, None])
        hg_dot_t = hg_dot_t.at[:, :, 2, 0].set(-alpha_dot_t[:, None])

        # nonlinear case
        dynamic_case = uvlm.prescribed_dynamic_solve(
            init_case=static_case, hg_t=hg_t, hg_dot_t=hg_dot_t
        )
        if plot:
            dynamic_case.plot(
                Path("./test_outputs/pitching_test_nonlinear_frozen_wake")
            )

        # linear case
        linear_model = uvlm.linearise(
            reference=static_case,
            wake_type="frozen",
            bound_upwash=False,
            wake_upwash=False,
            unsteady_force=True,
        )

        delta_zeta_b = dynamic_case.zeta_b - ArrayList(
            [zeta[None, ...] for zeta in linear_model.reference.zeta_b]
        )
        u_linear = AeroInputUnflattened(
            zeta_b=delta_zeta_b,
            zeta_b_dot=dynamic_case.zeta_b_dot,
            nu_b=None,
            nu_w=None,
        )

        # Run linear case with and without matrix form
        linear_case = linear_model.run(u_linear)

        if plot:
            linear_case.plot(
                Path(
                    f"./test_outputs/pitching_test_linear_frozen_wake_{'matrix' if use_matrix else 'operator'}"
                )
            )

        assert jnp.allclose(
            dynamic_case.gamma_b[0], linear_case.x_t_tot.gamma_b[0], atol=4e-2
        ), "Bound circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.gamma_w[0], linear_case.x_t_tot.gamma_w[0], atol=4e-2
        ), "Wake circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.f_steady[0], linear_case.y_t_tot.f_steady[0], atol=3e-1
        ), "Steady forces do not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.f_unsteady[0],
            cast(ArrayList, linear_case.y_t_tot.f_unsteady)[0],
            atol=1e-1,
        ), "Unsteady forces do not match between nonlinear and linear cases."

    @classmethod
    def test_linear_cosine_gust(cls, plot: bool = False, use_matrix: bool = False):
        flowfield = OneMinusCosineFlowField(
            TestLinearAero.u_inf,
            TestLinearAero.rho_inf,
            True,
            gust_length=2.0,
            gust_amplitude=0.4,
            gust_x0=jnp.array((-5.0, 0.0, 0.0)),
        )
        uvlm, static_case, hg0 = TestLinearAero.make_planar_wing(flowfield)

        hg_t = jnp.zeros((cls.n_tstep, *hg0.shape))
        hg_t = hg_t.at[...].set(hg0[None, ...])
        hg_dot_t = jnp.zeros_like(hg_t)

        # nonlinear case
        dynamic_case = uvlm.prescribed_dynamic_solve(
            init_case=static_case, hg_t=hg_t, hg_dot_t=hg_dot_t
        )
        if plot:
            dynamic_case.plot(Path("./test_outputs/gust_test_nonlinear"))

        # linear case
        linear_model = uvlm.linearise(
            reference=static_case,
            wake_type="prescribed",
            bound_upwash=True,
            wake_upwash=True,
            unsteady_force=True,
        )

        delta_zeta_b = dynamic_case.zeta_b - ArrayList(
            [zeta[None, ...] for zeta in linear_model.reference.zeta_b]
        )
        u_linear = AeroInputUnflattened(
            zeta_b=delta_zeta_b,
            zeta_b_dot=dynamic_case.zeta_b_dot,
            nu_b=None,
            nu_w=None,
        )

        # Run linear case with and without matrix form
        linear_case = linear_model.run(u_linear, flowfield=flowfield)

        if plot:
            linear_case.plot(
                Path(
                    f"./test_outputs/gust_test_linear_{'matrix' if use_matrix else 'operator'}"
                )
            )

        assert jnp.allclose(
            dynamic_case.gamma_b[0], linear_case.x_t_tot.gamma_b[0], atol=3e-1
        ), "Bound circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.gamma_w[0], linear_case.x_t_tot.gamma_w[0], atol=3e-1
        ), "Wake circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.zeta_w[0],
            cast(ArrayList, linear_case.x_t_tot.zeta_w)[0],
            atol=8e-2,
        ), "Wake grid coordinates do not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.f_steady[0], linear_case.y_t_tot.f_steady[0], atol=6e-1
        ), "Steady forces do not match between nonlinear and linear cases."
        assert jnp.allclose(
            dynamic_case.f_unsteady[0],
            cast(ArrayList, linear_case.y_t_tot.f_unsteady)[0],
            atol=8e-1,
        ), "Unsteady forces do not match between nonlinear and linear cases."
