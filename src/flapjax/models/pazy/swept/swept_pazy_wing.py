from typing import Literal

from jax import Array
from jax import numpy as jnp

from flapjax.aero.flowfields import ConstantFlowField, FlowField
from flapjax.coupled import CoupledAeroelastic
from flapjax.models.pazy.base import generate_generic_pazy_wing

# constant from provided data
from flapjax.models.pazy.swept.data.properties_10_deg import (
    SWEEP_10_LE,
    SWEEP_10_LE_CORRECTED,
    SWEEP_10_TE,
    SWEEP_10_TE_CORRECTED,
)
from flapjax.models.pazy.swept.data.properties_20_deg import (
    SWEEP_20,
)

DEFAULT_FLOWFIELD: FlowField = ConstantFlowField(
    u_inf=jnp.array((40.0, 0.0, 0.0)), rho=1.225, relative_motion=True
)
DEFAULT_AOA: Array = jnp.deg2rad(3.0)


def generate_swept_pazy_wing(
    m: int = 12,
    m_star: int = 120,
    sweep_angle: Literal[10, 20] = 10,
    tip_mass: (
        Literal["LE", "TE", "LE_CORRECTED", "TE_CORRECTED"] | None
    ) = "LE_CORRECTED",
    node_multiplier: int = 1,
    gravity: bool | Array = False,
    flowfield: FlowField = DEFAULT_FLOWFIELD,
    aoa: float | Array = DEFAULT_AOA,
    variable_disc_wake: bool = False,
    lumped_mass: bool = False,
    y_vector_override: Array | None = None,
) -> CoupledAeroelastic:
    match sweep_angle:
        case 10:
            match tip_mass:
                case "LE":
                    data = SWEEP_10_LE
                case "TE":
                    data = SWEEP_10_TE
                case "LE_CORRECTED":
                    data = SWEEP_10_LE_CORRECTED
                case "TE_CORRECTED":
                    data = SWEEP_10_TE_CORRECTED
                case None:
                    raise ValueError(
                        "No data available for 10 degree sweep with no tip mass"
                    )
                case _:
                    raise ValueError("Invalid tip_mass value")
        case 20:
            match tip_mass:
                case None:
                    data = SWEEP_20
                case _:
                    raise ValueError("Invalid tip_mass value")
        case _:
            raise ValueError("Invalid sweep_angle value")

    if y_vector_override is None and sweep_angle == 20:
        y_vector_override = jnp.array((0.0, 0.0, 1.0))

    return generate_generic_pazy_wing(
        m=m,
        m_star=m_star,
        node_multiplier=node_multiplier,
        gravity=gravity,
        flowfield=flowfield,
        aoa=aoa,
        data=data,
        sweep=None,  # if sweep were to be added here, it would be added on top of the sweep from data
        variable_disc_wake=variable_disc_wake,
        lumped_mass=lumped_mass,
        y_vector_override=y_vector_override,
    )
