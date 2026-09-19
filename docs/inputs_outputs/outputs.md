# Outputs

This page outlines the solver outputs: the primal solution objects returned by `static_solve` /
`dynamic_solve` at each of the three levels (structure, aero, coupled), modal and flutter results, gust frequency
responses, trim variables, and adjoint gradients. See [Inputs](inputs.md) for how to build the model that produces
these.

---

## 1. Solution objects at a glance

Structural problems return a `StructureCase`, aerodynamic problems return an `AeroCase`, and coupled aeroelastic
problems return an `AeroelasticCase` that wraps both through `.structure` and `.aero`.

---

## 2. Static, dynamic, and batched solutions

These classes are used for a range of problems:

- **Static solutions** — a single equilibrium configuration with no time history. Dynamic-only fields such as velocity
  and acceleration are zeroed out. Array shapes have no leading time axis, for example node coordinates `structure.x`
  are shape
  `(n_nodes, 3)` and bound circulation of a single surface `aero.gamma` is `(m, n)`.
- **Dynamic snapshot** — a single timestep with all dynamic fields populated. Data has the same shape as for static
  solutions.
- **Dynamic (batched)** — many timesteps stacked with a leading `n_tstep` axis on every array, for example node
  coordinates `structure.x` are shape `(n_tstep, n_nodes, 3)` and bound circulation of a single surface `aero.gamma` is
  `(n_tstep, m, n)`.
  We can extract a single snapshot from a batched trajectory using `case[i_ts]`, which returns a dynamic snapshot.

Use `.is_dynamic` and `.is_batched` to tell which type of solution you have at runtime.

```python
static_result = wing.static_solve(prescribed_dofs=tuple(range(6)))
static_result.is_dynamic()  # False
static_result.is_batched()  # False

dynamic_result = wing.dynamic_solve(init_case=static_result, n_tstep=100)
dynamic_result.is_batched()  # True
dynamic_result.is_dynamic()  # True

snapshot = dynamic_result[50]  # extract the timestep index 50 snapshot from the batched trajectory
snapshot.is_batched()  # False
snapshot.is_dynamic()  # True
```

---

## 3. Structural outputs — `StructureCase`

Note that the below shapes are for a single snapshot (static or dynamic). Batched solutions have a leading `n_tstep`
axis on every array.

### Kinematics

| Field    | Shape             | Meaning                                                                                                                                                                                          |
|----------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `x`      | `(n_nodes, 3)`    | Nodal global position.                                                                                                                                                                           |
| `rmat`   | `(n_nodes, 3, 3)` | Nodal rotation matrix, given relative to the reference configuration.                                                                                                                            |
| `eps`    | `(n_elem, 6)`     | Element strain vector; extensional/shear strain (0:3) and bending/torsional curvature (3:6).                                                                                                     |
| `hg`     | `(n_nodes, 4, 4)` | Nodal SE(3) homogeneous transform — position and orientation of each cross-section in the global frame.                                                                                          |
| `varphi` | `(n_nodes, 6)`    | Nodal configuration/twist vector: the $\mathfrak{se}(3)$ logarithm mapping the reference configuration to the current `hg`. This is the minimal coordinate system used internally by the solver. |
| `d`      | `(n_elem, 6)`     | Element relative configuration vector — the SE(3) twist between an element's two end nodes.                                                                                                      |

### Forces

| Field            | Shape                    | Meaning                                                                                                                 |
|------------------|--------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `f_ext_follower` | `(n_nodes, 6)` or `None` | Applied follower force/moment, if given from the solver inputs.                                                         |
| `f_ext_dead`     | `(n_nodes, 6)` or `None` | Applied dead force/moment, if given from the solver inputs.                                                             |
| `f_ext_aero`     | `(n_nodes, 6)` or `None` | Aerodynamic forcing projected onto the beam nodes.                                                                      |
| `f_grav`         | `(n_nodes, 6)` or `None` | Gravitational force/moment (`None` if no `gravity` vector was set on the structure).                                    |
| `f_int`          | `(n_nodes, 6)`           | Nodal force/moment contribution due to element stresses.                                                                |
| `f_elem`         | `(n_elem, 6)`            | Element-wise internal forces/moments.                                                                                   |
| `f_res`          | `(n_nodes, 6)`           | Residual (out-of-balance) nodal force/moment from the Newton solve. At prescribed DOFs, this equals the reaction force. |

### Dynamics

| Field        | Shape          | Meaning                                                                                 |
|--------------|----------------|-----------------------------------------------------------------------------------------|
| `v`          | `(n_nodes, 6)` | Nodal velocity.                                                                         |
| `v_dot`      | `(n_nodes, 6)` | Nodal acceleration.                                                                     |
| `a`          | `(n_nodes, 6)` | Generalised-$\alpha$ pseudo-acceleration auxiliary variable, used for time integration. |
| `f_iner_gyr` | `(n_nodes, 6)` | Inertial/gyroscopic forces.                                                             |

### Bookkeeping

| Field                            | Shape / type                  | Meaning                                                                                                                                                                            |
|----------------------------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `t`                              | scalar or `(n_tstep,)`        | Time.                                                                                                                                                                              |
| `i_ts`                           | `int` or `None`               | Timestep index.                                                                                                                                                                    |
| `prescribed_dofs` / `.free_dofs` | `tuple[int, ...]`             | Degree of freedom indices (of `n_nodes * 6`) that were removed vs. solved.                                                                                                         |      |
| `thrust`                         | `dict[str, Array]`            | Named thrust magnitude(s), matching the `thrust_nodes` passed to `BeamStructure`.                                                                                                  |
| `local`                          | `bool`                        | Frame flag — see below.                                                                                                                                                            |
| `constraint_data`                | `dict[str, dict[str, Array]]` | Per-named-constraint derived quantities, (see [Inputs](inputs.md#constraints)). For `MultibodyHinge` / `GroundedHinge` this is `{"angle": Array}` — the hinge rotation in radians. |

## 4. Frame convention: local vs. global

`StructureCase` (and `AeroelasticCase.structure`) carries a `local: bool` flag, `True` by default on every case
returned by a solve. When `local=True`, the following force/velocity/acceleration fields are expressed as components
in each node's local frame of reference:

- `f_ext_follower`
- `f_ext_dead`
- `f_ext_aero`
- `f_grav`
- `f_int`
- `f_res`
- `f_iner_gyr`
- `v`
- `v_dot`
- `a`

This is because the solver naturally expresses these values in their local frame. Call `.to_global()` on the solution
object to rotate these properties into the global/inertial frame, or `.to_local()` to reverse it. Nodal position and
orientation are always in the
global frame.

---

## 5. Aerodynamic outputs — `AeroCase`

Every field below is per-surface — an `ArrayList` of length `n_surf`, ordered the same as `surf_b_names`. For example,
to access the bound circulation of the first aerodynamic surface we use `AeroCase.gamma_b[0]`, The following property
shapes are given per surface:

| Field         | Shape (per surface) | Meaning                                                                                         |
|---------------|---------------------|-------------------------------------------------------------------------------------------------|
| `zeta_b`      | `(m+1, n+1, 3)`     | Bound-grid coordinates in the global frame..                                                    |
| `zeta_b_dot`  | `(m+1, n+1_n, 3)`   | Bound grid vertex velocities.                                                                   |
| `zeta_w`      | `(m_star, n+1, 3)`  | Wake grid vertex coordinates.                                                                   |
| `gamma_b`     | `(m, n)`            | Bound panel circulation strength.                                                               |
| `gamma_w`     | `(m_star, n)`       | Wake panel circulation strength.                                                                |
| `gamma_b_dot` | `(m, n)`            | Time derivative of bound circulation, driving the unsteady/added-mass force.                    |
| `f_steady`    | `(m+1, n+1, 3)`     | Steady (Kutta–Joukowski) force contribution per aerodynamic grid node.                          |
| `f_unsteady`  | `(m+1, n+1, 3)`     | Unsteady/apparent-mass force contribution per aerodynamic grid node.                            |
| `alpha`       | `(n,)`              | Per-spanwise-strip effective angle of attack.                                                   |
| `cs_ang`      | `dict[str, Array]`  | Control-surface deflection time history, `{name: ()}` snapshot or `{name: (n_tstep,)}` batched. |
| `cs_vel`      | `dict[str, Array]`  | Control-surface angular velocity time history.                                                  |
| `c`, `nc`     | `(m, n, 3)`         | Bound panel collocation points and normals.                                                     |
| `t`, `i_ts`   | As above            | Time / timestep index.                                                                          |

---

## 6. Coupled aeroelastic outputs — `AeroelasticCase`

`AeroelasticCase` is a thin wrapper: `.structure: StructureCase` and `.aero: AeroCase`, as returned by
`CoupledAeroelastic.static_solve` / `.dynamic_solve`.

```python
case = wing.static_solve(prescribed_dofs=jnp.arange(6))

case.structure.x[-1, :]  # tip position, shape (3)
case.aero.gamma_b[0]  # bound circulation on the first surface, shape (m, n)
case.structure.f_ext_aero  # aero forcing projected onto the beam
```

---

## 7. Trim outputs

`CoupledAeroelastic.trim(...)` returns `(AeroelasticCase, TrimVariables)` — the trimmed equilibrium
solution, and the trim variables that were solved for. These trim variables are given through:

| `TrimVariables` field | Type               | Meaning                                     |
|-----------------------|--------------------|---------------------------------------------|
| `cs_ang`              | `dict[str, Array]` | Trimmed control-surface deflection(s), rad. |
| `thrust`              | `dict[str, Array]` | Trimmed thrust value(s), N.                 |
| `trim_angles`         | `dict[str, Array]` | Rigid trim orientation, rad.                |
| `hinge_angle`         | `dict[str, Array]` | Trimmed `MultibodyHinge` rotation(s), rad.  |

---

## 8. Visualising results

We use the VTK file format for visualisation of the 3D structural and aerodynamic problems, which can be viewed in
ParaView. These files can be written by calling the `.plot(...)` methods on solution objects:

- `StructureCase.plot(...)` — Plot the deformed beam. Writes a single `.vtu` for a static case, or a `.pvd` +
  per-timestep `.vtu` series for a
  batched case..
- `AeroCase.plot(...)` — Plots the aerodynamic grid. Writes per-surface `.vts` grids for a single timestep, or a
  `.pvd` + per-timestep `.vts` for a batched case.
  series.
- `AeroelasticCase.plot(...)` — calls both of the above.

All VTK output (`.vtu`, `.vts`, `.pvd`) is intended for ParaView. Batched results are stitched into a single `.pvd`
collection so the whole time series opens and animates in one step.

It can often be handy to visualise the reference configuration when creating a model, to ensure that for example the
connectivity is correct. This can easily be done by using the `.get_reference()` method to get a solution object
representing the reference configuration, and then calling `.plot(...)` on that.

```python
# plot the reference configuration to paraview
wing.get_reference().plot("plot_dir/")  
```
