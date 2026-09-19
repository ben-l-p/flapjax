# Inputs

This page describes the inputs required to build a coupled aeroelastic model from scratch. See [Outputs](outputs.md)
for how to interpret and extract results once a model has been solved. A model is assembled in three stages:

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
    # required arguments
    num_nodes=...,
    connectivity=...,
    y_vector=...,
    # optional arguments
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

| Parameter      | Shape / type            | Description                                                                                                                                                        |
|----------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `num_nodes`    | `int`                   | Number of nodes in the beam assembly.                                                                                                                              |
| `connectivity` | `(n_elem, 2)` int       | Element connectivity — each row gives the two node indices forming an element. This also defines the order of the beam elements.                                   |
| `y_vector`     | `(n_elem, 3)` or `(3,)` | Out-of-plane reference vector for each element, used to define the cross-section frame. If a single `(3,)` vector is passed, it will be broadcast to all elements. |

### Optional parameters

| Parameter                     | Shape / type                                          | Default              | Description                                                                                                                                                                                                          |
|-------------------------------|-------------------------------------------------------|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `k_cs_index`                  | `(n_elem,)` int or `None`                             | `None`               | Index into the `k_cs` stiffness library for each element. `None` = every element uses library entry 0.                                                                                                               |
| `m_cs_index`                  | `(n_elem,)` int or `None`                             | `None`               | Index into the `m_cs` mass library for each element. `None` = every element uses library entry 0.                                                                                                                    |
| `m_lumped_index`              | `(n_lumped,)` int or `None`                           | `None`               | Node indices to attach a lumped mass to, in the same order as the `m_lumped` design variable. `None` = no lumped masses.                                                                                             |
| `gravity`                     | `(3,)` or `None`                                      | `None`               | Gravity vector in the global reference frame. `None` disables gravity.                                                                                                                                               |
| `thrust_nodes`                | `dict[str, int]` or `None`                            | `None`               | Named thrust points: maps a thrust name to the node index it acts on. `None` = no thrust.                                                                                                                            |
| `thrust_direction`            | `dict[str, (3,)]` or `None`                           | `None`               | Thrust direction vector for each name in `thrust_nodes`.                                                                                                                                                             |
| `optional_jacobians`          | `OptionalJacobians` or `None`                         | `None` (All `False`) | Flags to include additional Jacobian contributions in the Newton solver that are often omitted due to having a small contribution (dead-load stiffness, gravity stiffness, gyroscopic damping, geometric stiffness). |
| `relaxation_factor`           | `float`                                               | `1.0`                | Under-relaxation factor for the structural Newton update. `1.0` = no relaxation.                                                                                                                                     |
| `spectral_radius`             | `float`                                               | `0.9`                | Spectral radius of the generalised-alpha time integrator. `0.0` is highly damped, `1.0` is undamped.                                                                                                                 |
| `alpha_m`                     | `float`                                               | `0.0`                | Mass-proportional Rayleigh damping coefficient.                                                                                                                                                                      |
| `beta_k`                      | `float`                                               | `0.0`                | Stiffness-proportional Rayleigh damping coefficient.                                                                                                                                                                 |
| `struct_convergence_settings` | `ConvergenceSettings`                                 | See below            | Convergence tolerances and maximum iterations for the structural Newton loop.                                                                                                                                        |
| `constraints`                 | Dictionary of `constraint_name: Constraint` or `None` | `None`               | Introduce constraints into the system, such as hinges between bodies, or attach to a spring-dampter. (see [Constraints](#constraints)).                                                                              |

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

All fields, including `max_n_iter`, accept `None` to disable that criterion, but at least one tolerance or
`max_n_iter` must be set (non-`None`).

### Constraints

Constraints are passed to the `BeamStructure` constructor via the `constraints` argument as a `dict[str, Constraint]`
mapping a name to each constraint, and can be either soft (penalty-based) or hard (Lagrange-multiplier-based). The name
of each constraint is arbitrary, and is used for extracting data from the solution object such as hinge angles
(see [Outputs](outputs.md#bookkeeping)).

**Soft constraints** (subclasses of `SoftConstraint`) are imposed by adding stiffness/damping contributions to the
system, without needing Lagrange multipliers. They are generally more numerically stable, but do not enforce the
constraint exactly.

| Class                                               | Description                                                                                                                                                                                                     |
|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SpringDamper(node_index, k, hg_ref=None, c=None)`  | 6-DOF spring-damper at a node, relative to a fixed reference frame. `k` is `(6, 6)` stiffness, `c` is optional `(6, 6)` damping. If `hg_ref` is `None`, the node's initial pose is used as the reference frame. |
| `PrescribedMotion(node_index, k, hg_ref_t, c=None)` | Drive a node along a prescribed SE(3) trajectory `hg_ref_t` of shape `(n_tstep, 4, 4)` through a tracking spring-damper.                                                                                        |

**Hard constraints** (subclasses of `HardConstraint`) are imposed by adding Lagrange multipliers to the system, which
enforce the constraint near exactly.

| Class                                          | Description                                                                                                                                                          |
|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MultibodyHinge(node_i, node_j=None, *, axis)` | Hinge joint between two nodes allowing rotation only about `axis`. If `node_j=None`, a co-located node is automatically created which simplified model construction. |
| `GroundedHinge(node_i, *, axis, hg_ref=None)`  | Hinge pinning a node to a fixed point in space.                                                                                                                      |

For instance, the following would create a double pendulum, with the root node (index 0) a grounded hinge, and the
middle node (index 10) being the middle hinge, where both hinges are free to rotate about the y-axis.

```python
from flapjax.structure.constraints import MultibodyHinge, GroundedHinge

beam = BeamStructure(
    ...,
    constraints={
        "root_hinge": GroundedHinge(node_i=0, axis=jnp.array([0.0, 1.0, 0.0])),
        "mid_hinge": MultibodyHinge(node_i=10, axis=jnp.array([0.0, 1.0, 0.0])),
    },
)
```

---

## 2. Aerodynamics — `UVLM`

The aerodynamic model is an unsteady vortex lattice method with arbitrary numbers of lifting surfaces.

```python
from flapjax.aero.uvlm import UVLM
from flapjax.aero.data_structures import GridDiscretisation

uvlm = UVLM(
    # required arguments
    grid_shapes=...,
    dof_mapping=...,
    # optional arguments
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

Throughout the codebase and documentation, we will use `m`, `n`, and `m_star` to refer to the chordwise, spanwise, and
wake panel counts respectively, and `n_surf` to refer to the number of lifting surfaces, for shape annotations.

`GridDiscretisation` is a simple dataclass, where both of the following are equivalent:

```python
gd = GridDiscretisation(m=10, n=20, m_star=40)
gd_tuple = (10, 20, 40)
```

### Optional parameters

| Parameter                 | Type                         | Default              | Description                                                                                                                                                                                                                                            |
|---------------------------|------------------------------|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `variable_wake_disc`      | `bool`                       | `False`              | Allow non-uniform wake panel spacing (set via `delta_w` design variable). With a suitable `delta_w`, this can noticably reduce problem size by using larger wake panels far downstream.                                                                |
| `mirror_point`            | `(3,)` or `None`             | `None`               | Point on the (optional) aerodynamic symmetry plane. Both `mirror_point` and `mirror_normal` must be provided together, or both `None`.                                                                                                                 |
| `mirror_normal`           | `(3,)` or `None`             | `None`               | Normal vector of the symmetry plane.                                                                                                                                                                                                                   |
| `kernel`                  | `KernelFunction` or `None`   | `None` (Biot-Savart) | Custom Biot-Savart kernel for computing induced velocities. This should almost always be left to the Biot-Savart kernel.                                                                                                                               |
| `grid_func`               | `AeroGridFunction` or `None` | `None` (Identity)    | Function mapping `(zeta_b0, **cs_kwargs) -> zeta_b0_deflected` that takes the aerodynamic grid coordinates, as well as control surface deflections, and returns the deflected grid coordinates. This allows for arbitrary control surface definitions. |
| `free_wake`               | `bool`                       | `False`              | Include self-induced velocity in wake propagation. Generally does not have a large impact on results, and increases computational cost.                                                                                                                |
| `gamma_dot_relaxation`    | `float`                      | `0.7`                | Low-pass filter parameter for the time derivative of circulation, where larger values result in more damping.                                                                                                                                          |
| `include_unsteady_force`  | `bool`                       | `True`               | Include unsteady forcing contributions due to the circulation time derivative term.                                                                                                                                                                    |
| `batch_size`              | `int` or `None`              | `64`                 | Batch size for vectorised AIC computations. Larger values may be faster, but use more memory. `None` is equivalent to a full `vmap`.                                                                                                                   |
| `polars`                  | Sequence or `None`           | `None`               | Per-surface, per-strip tabulated airfoil polars for sectional force correction. `None` disables polar correction for all surfaces; otherwise each entry is either `None` or a sequence of `n` callables mapping `alpha -> (cl, cd, cm)`.               |
| `polar_circulation_scale` | `float`                      | `0.0`                | Factor in `[0, 1]` controlling how much of the polar lift correction is applied to the bound circulation before it is convected into the wake.                                                                                                         |

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
    # required
    coords=...,
    k_cs=...,
    m_cs=...,
    m_lumped=None,
    dt=...,
    flowfield=...,
    x0_aero=...,
    # optional
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
| `m_cs`              | `(n_entry, 6, 6)` or `(6, 6)` or `None` | Cross-section mass matrices. `None` sets zero element mass. Entries ordered as `[m, m, m, I_xx, I_yy, I_zz]` on the diagonal for isotropic sections.                                                                                                            |
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

# geometry and discretisation
n_nodes = 21  # number of beam nodes - node that this will result in (n_nodes - 1) spanwise panels
n_elem = n_nodes - 1
span = 5.0
chord = 1.0
ea = 0.25  # elastic axis at quarter-chord
m, m_star = 8, 20  # chordwise and wake panel discretisations

# create structure
# simple connectivity for serial connection of elements, ((0, 1), (1, 2), (2, 3), ...)
connectivity = jnp.column_stack([jnp.arange(n_elem), jnp.arange(1, n_nodes)])
beam = BeamStructure(
    num_nodes=n_nodes,
    connectivity=connectivity,
    y_vector=jnp.array([0.0, 0.0, 1.0]),
    # we orient the beam to point in the +y direction, which makes the z-axis the out-of-plane direction
    gravity=jnp.array([0.0, 0.0, -9.81]),  # gravity is passed as a vector in the global frame
)

# create aerodynamics
uvlm = UVLM(
    grid_shapes=[GridDiscretisation(m=m, n=n_elem, m_star=m_star)],
    # easy mapping for a single lifting surface - strip 0 maps to node 0, strip 1 maps to node 1, etc
    dof_mapping=jnp.arange(n_nodes),
    # add an aerodynamic mirror plane at the root - this plane is defined with a point and a normal
    mirror_point=jnp.zeros(3),
    mirror_normal=jnp.array([0.0, 1.0, 0.0]),
)

# create coupled system
wing = CoupledAeroelastic(beam, uvlm)

# create the design variables for the system
# beam coordinates for straight beam along the y-axis, with the root at the origin
coords = jnp.zeros((n_nodes, 3)).at[:, 1].set(jnp.linspace(0, span, n_nodes))

# cross-section stiffness and mass libraries - all elements use the same properties and so both are [6, 6] matrices
# stiffness for a beam with shear deformation: (EA GAy GAz GJ EIy EIz)
k_cs = jnp.diag(jnp.array([1e6, 1e6, 1e6, 4e3, 4e3, 4e3]))
# properties per unit length: (m, m, m, Ixx, Iyy, Izz)
m_cs = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

u_inf = jnp.array([10.0, 0.0, 1.0])  # freestream velocity vector - positive angle of attack
dt = chord / (jnp.linalg.norm(u_inf) * m)  # timestep length based on CFL condition for chordwise panels

# the freestream is defined with a `ConstantFlowField` object, which is a flow constant in space and time. The 
# `relative_motion` flag is set to `True` to indicate moving air, stationary wing
flowfield = ConstantFlowField(u_inf=u_inf, rho=1.225, relative_motion=True)

# create the aerodynamic grid coordinates using helper function, (m+1, n+1, 3) shape
# note these coordinates are given relative to the beam coordinate
grid = make_rectangular_grid(m=m, n=n_elem, chord=chord, ea=ea)

# set the design variables for the coupled system
wing.set_design_variables(
    coords=coords,
    k_cs=k_cs,
    m_cs=m_cs,
    m_lumped=None,
    dt=dt,
    flowfield=flowfield,
    x0_aero=grid,
)

# solve a static problem
# we clamp the wing by prescribing the first 6 DOFs (0, 1, 2, 3, 4, 5), which eliminates them from the system to be 
# solved and enforces zero displacement. Linear and rotational degrees of freedom are given in order of (x, y, z, rx, ry, 
# rz), with increasing node number, so the next node has DOFs (6, 7, 8,  9, 10, 11) etc.
static_result = wing.static_solve(
    prescribed_dofs=tuple(range(6)),
)

# solve a dynamic problem (as this is a constant flowfield, the dynamic solution will be the same as the static solution)
# pass the static solution as the initial condition for the dynamic problem, and set the number of timesteps
dynamic_result = wing.dynamic_solve(
    init_case=static_result,
    n_tstep=100,
)
```