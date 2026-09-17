from collections.abc import Sequence

from jax import Array
from jax import numpy as jnp
from jax.scipy.linalg import block_diag

from flapjax.structure import BeamStructure
from flapjax.structure.constraints import GroundedHinge, MultibodyHinge


def generate_double_pendulum(
    length: float = 1.0,
    n_nodes_per_segment: int = 4,
    hinge_axis: Array | Sequence[float] = (0.0, 1.0, 0.0),
    gravity: Array | Sequence[float] = (0.0, 0.0, -9.81),
    k_axial: float = 1e6,
    k_bending: float = 1e4,
    k_torsional: float = 1e4,
    mass_per_length: float = 5.0,
    rotary_inertia: float = 0.1,
) -> tuple[BeamStructure, GroundedHinge, MultibodyHinge]:
    r"""
    Generate a double-pendulum model, consisting of two flexible beams, connected by a hinge, with the first beam
    mounted to a fixed point with a hinge.
    :param length: Length of each pendulum segment.
    :param n_nodes_per_segment: Nodes per segment (minimum 2).
    :param hinge_axis: Hinge rotation axis, ``(3, )``.
    :param gravity: Gravity vector, ``(3, )``.
    :param k_axial: Axial and shear stiffness of each segment.
    :param k_bending: Bending stiffness of each segment.
    :param k_torsional: Torsional stiffness of each segment.
    :param mass_per_length: Translational mass per unit length.
    :param rotary_inertia: Rotary inertia per unit length.
    :return: ``(beam, root_hinge, mid_hinge)`` tuple.
    """
    n = n_nodes_per_segment
    n_nodes = 2 * n - 1
    n_elem_per_seg = n - 1

    # connectivity - note this is the same as for a single beam
    # the solver automatically updates the changes from the multibody constraints
    conn = jnp.stack(
        (jnp.arange(2 * n_elem_per_seg), jnp.arange(1, 2 * n_elem_per_seg + 1)), axis=1
    )

    root_hinge = GroundedHinge(node_i=0, axis=jnp.array(hinge_axis))
    mid_hinge = MultibodyHinge(node_i=n - 1, node_j=None, axis=jnp.array(hinge_axis))

    beam = BeamStructure(
        num_nodes=n_nodes,
        connectivity=conn,
        y_vector=jnp.array((0.0, 0.0, 1.0)),
        constraints={"root_hinge": root_hinge, "mid_hinge": mid_hinge},
        gravity=jnp.array(gravity),
        spectral_radius=0.9,
    )

    x = jnp.linspace(0.0, 2 * length, 2 * n - 1)
    coords = jnp.stack((x, jnp.zeros(n_nodes), jnp.zeros(n_nodes)), axis=1)

    k_cs = jnp.diag(
        jnp.array((k_axial, k_axial, k_axial, k_torsional, k_bending, k_bending))
    )
    m_cs = block_diag(mass_per_length * jnp.eye(3), rotary_inertia * jnp.eye(3))

    beam.set_design_variables(coords=coords, k_cs=k_cs, m_cs=m_cs)
    return beam, root_hinge, mid_hinge
