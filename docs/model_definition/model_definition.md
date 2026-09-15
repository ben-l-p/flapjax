# Model definition

This page describes the inputs required to build a coupled aeroelastic model from scratch. A model is assembled in three
stages:

1. **Instantiate** the structure (`BeamStructure`) and aerodynamics (`UVLM`) with fixed, non-design parameters. These
   define the "layout" of the problem, such as the number of nodes, element connectivity, and aerodynamic grid shapes.
2. **Couple** them into a `CoupledAeroelastic` object.
3. **Set design variables** — set the continuous mutable physical quantities (coordinates, stiffness, flow conditions,
   …) that define a specific case, and can later be differentiated with respect to.

This approach to split the problem into non-design and design parameters is a necessity for differentiable programming.

---

## 1. Structure — `BeamStructure`

The structural beam model. All constructor arguments are parameters that remain fixed once the object is created.

```python
from jax import numpy as jnp
from flapjax.structure import BeamStructure

beam = BeamStructure(
    num_nodes=...,
    connectivity=...,
    y_vector=...,
    # --- optional ---
    k_cs_index=None,
    m_cs_index=None,
    m_lumped_index=None,
    gravity=None,
    thrust_nodes=None,
    thrust_direction=None,
    optional_jacobians=None,
    relaxation_factor=1.0,
    spectral_radius=0.9,
    alpha_m=0.0,
    beta_k=0.0,
    struct_convergence_settings=...,
    constraints=None,
)
```

### Required parameters

| Parameter      | Shape / type            | Description                                                                                                                                                                                                          |
|----------------|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `num_nodes`    | `int`                   | Number of nodes in the beam.                                                                                                                                                                                         |
| `connectivity` | `(n_elem, 2)` int       | Element connectivity — each row gives the two node indices forming an element.                                                                                                                                       |
| `y_vector`     | `(n_elem, 3)` or `(3,)` | Out-of-plane reference vector for each element, used to define the cross-section frame. If a single `(3,)` vector is passed, it will be broadcast to all elements. Must not be collinear with the element direction. |

### Optional parameters

| Parameter                     | Shape / type           | Default     | Description                                                                                                                                                                                                          |
|-------------------------------|------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `k_cs_index`                  | `(n_elem,)` int        | All zeros   | Index into the stiffness library for each element. Allows different elements to use different cross-section stiffness matrices.                                                                                      |
| `m_cs_index`                  | `(n_elem,)` int        | All zeros   | Index into the mass library for each element.                                                                                                                                                                        |
| `m_lumped_index`              | `(n_lumped,)` int      | `None`      | Node indices at which lumped masses are attached. The ordering corresponds to the rows of the `m_lumped` design variable.                                                                                            |
| `gravity`                     | `(3,)`                 | `None`      | Gravity vector in the global frame. Pass `None` to disable gravity.                                                                                                                                                  |
| `thrust_nodes`                | `dict[str, int]`       | `None`      | Named thrust application nodes, e.g. `{"left_engine": 5}`.                                                                                                                                                           |
| `thrust_direction`            | `dict[str, (3,)]`      | `None`      | Direction of thrust at each named node. Keys must match `thrust_nodes`.                                                                                                                                              |
| `optional_jacobians`          | `OptionalJacobians`    | All `False` | Flags to include additional Jacobian contributions in the Newton solver that are often omitted due to having a small contribution (dead-load stiffness, gravity stiffness, gyroscopic damping, geometric stiffness). |
| `relaxation_factor`           | `float`                | `1.0`       | Under-relaxation factor for the structural Newton update. `1.0` = no relaxation.                                                                                                                                     |
| `spectral_radius`             | `float`                | `0.9`       | Spectral radius of the generalised-alpha time integrator. `0.0` is highly damped, `1.0` is undamped.                                                                                                                 |
| `alpha_m`                     | `float`                | `0.0`       | Mass-proportional Rayleigh damping coefficient.                                                                                                                                                                      |
| `beta_k`                      | `float`                | `0.0`       | Stiffness-proportional Rayleigh damping coefficient.                                                                                                                                                                 |
| `struct_convergence_settings` | `ConvergenceSettings`  | See below   | Convergence tolerances and maximum iterations for the structural Newton loop.                                                                                                                                        |
| `constraints`                 | Constraint or sequence | `None`      | Soft and/or hard constraints (see [Constraints](#constraints)).                                                                                                                                                      |

### Convergence settings

Both the structural loop and the FSI loop use `ConvergenceSettings`:

```python
from flapjax.utils.data_structures import ConvergenceSettings

settings = ConvergenceSettings(
    max_n_iter=25,
    rel_disp_tol=1e-5,
    abs_disp_tol=1e-7,
    rel_force_tol=1e-6,
    abs_force_tol=1e-8,
)
```

All tolerance fields accept `None` to disable that criterion, but at least one tolerance or the maximum iteration count
`max_n_iter` must be set.

### Constraints

Constraints are passed to the `BeamStructure` constructor via the `constraints` argument and can be either soft
(penalty-based) or hard (Lagrange-multiplier-based). A single constraint, or an arbitrary sequence of constraints, can
be passed.

**Soft constraints** (subclasses of `SoftConstraint`):

| Class                                               | Description                                                                                                                                                                                                     |
|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SpringDamper(node_index, k, hg_ref=None, c=None)`  | 6-DOF spring-damper at a node, relative to a fixed reference frame. `k` is `(6, 6)` stiffness, `c` is optional `(6, 6)` damping. If `hg_ref` is `None`, the node's initial pose is used as the reference frame. |
| `PrescribedMotion(node_index, k, hg_ref_t, c=None)` | Drive a node along a prescribed SE(3) trajectory `hg_ref_t` of shape `(n_tstep, 4, 4)` through a tracking spring-damper.                                                                                        |

**Hard constraints** (subclasses of `HardConstraint`):

| Class                                          | Description                                                                                                                      |
|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `MultibodyHinge(node_i, node_j=None, *, axis)` | Hinge joint between two nodes allowing rotation only about `axis`. If `node_j=None`, a co-located node is automatically created. |
| `GroundedHinge(node_i, *, axis, hg_ref=None)`  | Hinge pinning a node to a fixed point in space.                                                                                  |

For instance, the following would create a double pendulum, with the root node (index 0) a grounded hinge, and the
middle node (index 10) being the middle hinge, where both hinges are free to rotate about the y-axis.

```python
from flapjax.structure.constraints import MultibodyHinge, GroundedHinge

beam = BeamStructure(
    ...,
    constraints=[
        GroundedHinge(node_i=0, axis=jnp.array([0.0, 1.0, 0.0])),
        MultibodyHinge(node_i=10, axis=jnp.array([0.0, 1.0, 0.0])),
    ],
)
```

---

## 2. Aerodynamics — `UVLM`

The aerodynamic model is an unsteady vortex lattice method with arbitrary numbers of lifting surfaces.

```python
from flapjax.aero.uvlm import UVLM
from flapjax.aero.data_structures import GridDiscretisation

uvlm = UVLM(
    grid_shapes=...,
    dof_mapping=...,
    # --- optional ---
    variable_wake_disc=False,
    mirror_point=None,
    mirror_normal=None,
    kernel=None,
    grid_func=None,
    free_wake=False,
    gamma_dot_relaxation=0.7,
    include_unsteady_force=True,
    batch_size=64,
    polars=None,
    polar_circulation_scale=0.0,
)
```

### Required parameters

| Parameter     | Shape / type                                                | Description                                                                       |
|---------------|-------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `grid_shapes` | Sequence of `GridDiscretisation` or `(m, n, m_star)` tuples | Panel counts for each surface: `m` chordwise, `n` spanwise, `m_star` wake panels. |
| `dof_mapping` | `(n_surf,)(n+1,)` int                                       | Per-surface mapping from spanwise aerodynamic stations to beam node indices.      |

`GridDiscretisation` is a simple dataclass, where both of the following are equivalent:

```python
gd = GridDiscretisation(m=10, n=20, m_star=40)
gd_tuple = (10, 20, 40)
```

### Optional parameters

| Parameter                 | Type               | Default     | Description                                                                                                                                                                                                                                            |
|---------------------------|--------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `variable_wake_disc`      | `bool`             | `False`     | Allow non-uniform wake panel spacing (set via `delta_w` design variable).                                                                                                                                                                              |
| `mirror_point`            | `(3,)`             | `None`      | Point on the aerodynamic symmetry plane. Both `mirror_point` and `mirror_normal` must be provided together, or both `None`.                                                                                                                            |
| `mirror_normal`           | `(3,)`             | `None`      | Normal vector of the symmetry plane.                                                                                                                                                                                                                   |
| `kernel`                  | `KernelFunction`   | Biot-Savart | Custom Biot-Savart kernel for computing induced velocities.                                                                                                                                                                                            |
| `grid_func`               | `AeroGridFunction` | Identity    | Function mapping `(zeta_b0, **cs_kwargs) -> zeta_b0_deflected` that takes the aerodynamic grid coordinates, as well as control surface deflections, and returns the deflected grid coordinates. This allows for arbitrary control surface definitions. |
| `free_wake`               | `bool`             | `False`     | Include self-induced velocity in wake propagation. Generally does not have a large impact on results, and increases computational cost.                                                                                                                |
| `gamma_dot_relaxation`    | `float`            | `0.7`       | Low-pass filter parameter for the time derivative of circulation, where larger values result in more damping.                                                                                                                                          |
| `include_unsteady_force`  | `bool`             | `True`      | Include unsteady forcing contributions due to the circulation time derivative term.                                                                                                                                                                    |
| `batch_size`              | `int` or `None`    | `64`        | Batch size for vectorised AIC computations. Larger values may be faster, but use more memory. `None` is equivalent to a full `vmap`.                                                                                                                   |
| `polars`                  | Sequence           | `None`      | Per-surface, per-strip tabulated airfoil polars for sectional force correction. Each entry is either `None` or a sequence of `n` callables mapping `alpha -> (cl, cd, cm)`.                                                                            |
| `polar_circulation_scale` | `float`            | `0.0`       | Factor in `[0, 1]` controlling how much of the polar lift correction is applied to the bound circulation before it is convected into the wake.                                                                                                         |

---

## 3. Coupling — `CoupledAeroelastic`

Wrap the structure and aero into a coupled system:

```python
from flapjax.coupled import CoupledAeroelastic

wing = CoupledAeroelastic(
    structure=beam,
    aero=uvlm,
    fsi_convergence_settings=...,  # optional, defaults provided
)
```

The FSI convergence settings control the Picard iteration with Aitken relaxation used by `static_solve`. The default is:

```python
ConvergenceSettings(
    max_n_iter=25,
    rel_disp_tol=1e-3,
    abs_disp_tol=1e-5,
    rel_force_tol=1e-3,
    abs_force_tol=1e-5,
)
```

---

## 4. Design variables — `set_design_variables`

Design variables are the mutable continuous physical quantities that describe a particular configuration. They are set
after
construction and are the quantities that the adjoint machinery can differentiate with respect to. Whilst this here sets
the design variables for an aeroelastic system, both the structure and aero objects also have their own
`set_design_variables` methods that can be used independently.

```python
wing.set_design_variables(
    coords=...,
    k_cs=...,
    m_cs=...,
    m_lumped=None,
    dt=...,
    flowfield=...,
    x0_aero=...,
    # --- optional ---
    delta_w=None,
    thrust_reference=None,
    orientation_euler=None,
    cs_angles_reference=None,
)
```

### Structure design variables

| Parameter           | Shape                                   | Description                                                                                                                                                                                                                                                     |
|---------------------|-----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `coords`            | `(n_nodes, 3)`                          | Beam node coordinates in the undeformed reference configuration.                                                                                                                                                                                                |
| `k_cs`              | `(n_entry, 6, 6)` or `(6, 6)`           | Cross-section stiffness matrix library. A single `(6, 6)` is broadcast to all elements. Multiple entries are indexed by `k_cs_index` along the leading axis. For a beam without cross couplings, each element is given as `diag([EA, GAy, GAz, GJ, EIy, EIy])`. |
| `m_cs`              | `(n_entry, 6, 6)` or `(6, 6)` or `None` | Cross-section mass matrices. `None` sets zero mass (valid only without gravity). Entries ordered as `[m, m, m, I_xx, I_yy, I_zz]` on the diagonal for isotropic sections.                                                                                       |
| `m_lumped`          | `(n_lumped, 6, 6)` or `None`            | Lumped mass matrices at the nodes specified by `m_lumped_index`.                                                                                                                                                                                                |
| `orientation_euler` | `(3,)` or `None`                        | Euler angles (radians, z-y-x order) to rotate the reference configuration about the origin. Can be used to apply angle of incidence.                                                                                                                            |
| `thrust_reference`  | `dict[str, (1,)]` or `None`             | Reference thrust magnitude at each named thrust node. Must use the same keys as the thrust nodes in the model.                                                                                                                                                  |

### Aero design variables

| Parameter             | Shape                         | Description                                                                                                                                      |
|-----------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `dt`                  | scalar                        | Time step length. A common choice is `chord / (u_inf * M)` for `M` chordwise panels                                                              |u_inf| * m)`. |
| `flowfield`           | `FlowField`                   | Flow field object defining background velocity as a function of space and time.                                                                  |
| `x0_aero`             | `(n_surf,)(m+1, n+1, 3)`      | Local aerodynamic grid coordinates for each surface. Can be a single `Array` for one surface or an `ArrayList`/`Sequence` for multiple surfaces. |
| `delta_w`             | per-surface `Array` or `None` | Non-uniform wake segment lengths. Only used when `variable_wake_disc=True`.                                                                      |
| `cs_angles_reference` | `dict[str, Array]` or `None`  | Reference control-surface angles.                                                                                                                |

### Flow fields

Flow field objects define the background velocity at any point in space and time.

```python
from flapjax.aero.flowfields import ConstantFlowField, OneMinusCosineFlowField
```

**`ConstantFlowField`** — uniform steady flow:

| Parameter         | Type    | Description                                                                             |
|-------------------|---------|-----------------------------------------------------------------------------------------|
| `u_inf`           | `(3,)`  | Freestream velocity vector.                                                             |
| `rho`             | `float` | Air density.                                                                            |
| `relative_motion` | `bool`  | `True` if the air moves past a fixed wing; `False` if the wing moves through still air. |

**`OneMinusCosineFlowField`** — steady flow plus a 1-cos gust:

| Parameter                  | Type             | Description                                                               |
|----------------------------|------------------|---------------------------------------------------------------------------|
| `u_inf`                    | `(3,)`           | Base freestream velocity.                                                 |
| `rho`                      | `float`          | Air density.                                                              |
| `relative_motion`          | `bool`           | As above.                                                                 |
| `gust_length`              | `float`          | Gust half-wavelength.                                                     |
| `gust_amplitude`           | `float`          | Peak gust velocity.                                                       |
| `gust_travel_direction`    | `(3,)` or `None` | Direction the gust travels. Defaults to freestream direction.             |
| `gust_amplitude_direction` | `(3,)` or `None` | Direction the gust acts. Defaults to `(0, 0, 1)`.                         |
| `gust_x0`                  | `(3,)` or `None` | Spatial origin of the gust leading edge at `t=0`. Defaults to the origin. |

### Aerodynamic grid helper

For simple rectangular planforms, a utility function generates the local grid coordinates:

```python
from flapjax.aero.utils import make_rectangular_grid

grid = make_rectangular_grid(m=10, n=20, chord=1.0, ea=0.25)
# returns shape (m+1, n+1, 3)
```

| Parameter | Type    | Description                                                                         |
|-----------|---------|-------------------------------------------------------------------------------------|
| `m`       | `int`   | Chordwise panels.                                                                   |
| `n`       | `int`   | Spanwise panels.                                                                    |
| `chord`   | `float` | Chord length.                                                                       |
| `ea`      | `float` | Elastic axis location as a fraction of chord (0 = leading edge, 1 = trailing edge). |

---

## Complete example

A minimal cantilever wing with constant spanwise properties:

```python
import jax
import jax.numpy as jnp
from flapjax.structure import BeamStructure
from flapjax.aero.uvlm import UVLM
from flapjax.aero.data_structures import GridDiscretisation
from flapjax.aero.flowfields import ConstantFlowField
from flapjax.aero.utils import make_rectangular_grid
from flapjax.coupled import CoupledAeroelastic

# --- geometry ---
n_nodes = 21
n_elem = n_nodes - 1
span = 5.0
chord = 1.0
ea = 0.25  # elastic axis at quarter-chord
m, m_star = 8, 20  # chordwise and wake panels

# --- structure ---
connectivity = jnp.column_stack([jnp.arange(n_elem), jnp.arange(1, n_nodes)])  # nodes in serial order
beam = BeamStructure(
    num_nodes=n_nodes,
    connectivity=connectivity,
    y_vector=jnp.array([0.0, 0.0, 1.0]),
    gravity=jnp.array([0.0, 0.0, -9.81]),
)

# --- aerodynamics ---
uvlm = UVLM(
    grid_shapes=[GridDiscretisation(m=m, n=n_elem, m_star=m_star)],
    dof_mapping=jnp.arange(n_nodes),
    mirror_point=jnp.zeros(3),
    mirror_normal=jnp.array([0.0, 1.0, 0.0]),
)

# --- coupled system ---
wing = CoupledAeroelastic(beam, uvlm)

# --- design variables ---
coords = jnp.zeros((n_nodes, 3)).at[:, 1].set(jnp.linspace(0, span, n_nodes))

k_cs = jnp.diag(jnp.array([1e6, 1e6, 1e6, 4e2, 4e2, 4e2]))
m_cs = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

u_inf = jnp.array([10.0, 0.0, 1.0])
dt = chord / (jnp.linalg.norm(u_inf) * m)
flowfield = ConstantFlowField(u_inf=u_inf, rho=1.225, relative_motion=True)
grid = make_rectangular_grid(m=m, n=n_elem, chord=chord, ea=ea)

wing.set_design_variables(
    coords=coords,
    k_cs=k_cs,
    m_cs=m_cs,
    m_lumped=None,
    dt=dt,
    flowfield=flowfield,
    x0_aero=grid,
)

# --- solve ---
# Static: clamp root (first 6 DOFs)
static_result = wing.static_solve(
    prescribed_dofs=jnp.arange(6),
)

# Dynamic: 100 time steps from the static equilibrium
dynamic_result = wing.dynamic_solve(
    init_case=static_result,
    n_tstep=100,
)
```