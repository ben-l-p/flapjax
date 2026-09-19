from typing import Literal

import pytest
from jax import Array
from jax import numpy as jnp

from flapjax.aero.flowfields import ConstantFlowField
from flapjax.models.pazy.straight.pazy_wing import generate_pazy_wing
from flapjax.models.pazy.swept.swept_pazy_wing import generate_swept_pazy_wing


def _swept_modes(
    tip_mass: Literal["LE", "TE", "LE_CORRECTED", "TE_CORRECTED"] | None,
    sweep_angle: Literal[10, 20] = 10,
    n_modes: int = 3,
    lumped_mass: bool = False,
    node_multiplier: int = 2,
) -> Array:
    wing = generate_swept_pazy_wing(
        flowfield=ConstantFlowField(
            u_inf=jnp.array([0.0, 0.0, 0.0]),
            rho=1.225,
            relative_motion=True,
        ),
        tip_mass=tip_mass,  # type: ignore[arg-type]
        aoa=jnp.deg2rad(0.0),
        m=6,
        m_star=0,
        node_multiplier=node_multiplier,
        sweep_angle=sweep_angle,
        lumped_mass=lumped_mass,
    )

    reference = wing.reference_configuration(prescribed_dofs=tuple(range(6)))

    freqs, *_ = wing.structure.modal(
        case=reference.structure,
        n_modes=n_modes,
    )
    return freqs


SWEPT_CASES = [
    pytest.param(
        "LE",
        10,
        (4.31, 28.9, 37.5),
        (1.1e-2, 1e-2, 0.1),
        False,
        2,
        id="sweep_10_le",
    ),
    pytest.param(
        "LE_CORRECTED",
        10,
        (4.31, 28.9, 37.5),
        (1.1e-2, 1e-2, 2e-2),
        False,
        2,
        id="sweep_10_le_corrected",
    ),
    pytest.param(
        "TE",
        10,
        (4.27, 27.4, 38.7),
        (1e-2, 5e-2, 2e-2),
        False,
        2,
        id="sweep_10_te",
    ),
    pytest.param(
        "TE_CORRECTED",
        10,
        (4.27, 27.4, 38.7),
        (1e-2, 1e-2, 2e-2),
        False,
        2,
        id="sweep_10_te_corrected",
    ),
    # pytest.param(
    #     None,
    #     20,
    #     (4.568, 28.627, 44.534),
    #     (1e-2, 1e-2, 1e-2),
    #     True,
    #     1,
    #     id="sweep_20",
    # ),
]


class TestPazyModal:
    @staticmethod
    def test_straight_modes():
        r"""
        Check the first 5 modes of the Pazy model, and that they are close to literature. Using the model from
        https://github.com/UM-A2SRL/AePW3-LDWG/tree/main/00_Models/01_Pazy_Technion/02_Models_Beam and comparing with
        modes presented in "Collaborative Pazy Wing Analyses for the Third Aeroelastic Prediction Workshop".
        """

        wing = generate_pazy_wing(
            flowfield=ConstantFlowField(
                u_inf=jnp.array([0.0, 0.0, 0.0]),
                rho=1.225,
                relative_motion=True,
            ),
            aoa=jnp.deg2rad(0.0),
            m=6,
            m_star=0,
            node_multiplier=2,
            skin=True,
            variable_disc_wake=False,
            sweep=0.0,
        )

        reference = wing.reference_configuration(prescribed_dofs=tuple(range(6)))

        freqs, *_ = wing.structure.modal(
            case=reference.structure,
            n_modes=5,
        )

        literature_freqs = jnp.array((4.19, 28.47, 40.74, 82.39, 107.78))  # hz

        # torsion mode has a larger discrepancy but this is expected - we apply per-mode tolerancing from 1-4%
        tolerances = jnp.array((1e-2, 2e-2, 4e-2, 2e-2, 1e-2))

        error = jnp.abs((freqs / literature_freqs) - 1.0)

        assert jnp.all(error < tolerances), (
            "Pazy undeformed structural frequencies do not match literature frequencies"
        )

    @staticmethod
    @pytest.mark.parametrize(
        "tip_mass, sweep_angle, literature_freqs, tolerances, lumped_mass, node_multiplier",
        SWEPT_CASES,
    )
    def test_swept_modes(
        tip_mass: Literal["LE", "TE", "LE_CORRECTED", "TE_CORRECTED"] | None,
        sweep_angle: Literal[10, 20],
        literature_freqs: tuple[float, ...],
        tolerances: tuple[float, ...],
        lumped_mass: bool,
        node_multiplier: int,
    ):
        r"""
        Check the first 3 modes of a swept Pazy model against literature/FE reference values.
        See the model data for the source of the data.
        """
        freqs = _swept_modes(
            tip_mass=tip_mass,
            sweep_angle=sweep_angle,
            lumped_mass=lumped_mass,
            node_multiplier=node_multiplier,
        )

        error = jnp.abs((freqs / jnp.array(literature_freqs)) - 1.0)

        assert jnp.all(error < jnp.array(tolerances)), (
            "Pazy undeformed structural frequencies do not match literature frequencies"
        )
