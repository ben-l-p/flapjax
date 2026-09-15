# flapjax

**flapjax** (FLexible Adjoint-enabled Panel code in JAX) is a nonlinear aeroelastic solver build using
Google JAX. It couples:

- **Aerodynamics** - Unsteady Vortex Lattice Method (UVLM)
- **Structure** - Lie-group nonlinear beam formulation

This allows for accurate simulation of highly flexible aircraft configurations. Capabilities for simulation include:

- **Static aeroelastic analysis** for finding the equilibrium configuration
- **Dynamic aeroelastic analysis** for time-domain response to gusts and control inputs
- **Linearised aeroelastic analysis** for structural modal, and aeroelastic stability and flutter analysis
- **Adjoint-based gradients** for efficient sensitivity computation of static, dynamic and linearised systems, enabling
  gradient-based optimisation

There are many advantages for using a fully JAX-based framework, including:

- **Fast analysis** using JAX's just-in-time compilation
- **Parallelisation** using JAX's vectorisation and batching capabilities, which allows for parallelising across
  multiple cases, as well as within a single case. This supports execution on CPU and GPU hardware.
- **Automatic differentiation** of Python functions, which allows for an elegant implementation of adjoint and
  linearised systems.

It is advised that users have an understanding of key JAX principles, including JIT compilation, vectorisation, and
automatic differentiation [JAX documentation](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html). This can
often result in unusual coding patterns due to the absense of conventional control logic.

This framework is designed to be modular, and allows for very flexible workflows. It makes extensive use of
object-oriented principles with a very "pythonic" interface. It makes full use of Python type hinting to best inform
users of the expected input and output types for each function.

<div class="grid cards" markdown>

- :material-school: **[Modelling](model_definition/model_definition.md)**
  How to construct an aeroelastic model.

- :material-school: **[Tutorials](tutorials/index.md)**
  Example-based guides.

- :material-book-open-variant: **[Reference](reference/index.md)**
  API documentation.

- :material-lightbulb: **[Theory](theory/index.md)**
  Theoretical background behind the aeroelastic modelling.

</div>

## Installation

Installation is simple, supporting pip installation. It is recommended to install in a virtual environment, with ``uv``
being the recommended package manager. The following commands will install the package with all dependencies, including
optional dependencies for development.

Recommended installation with ``uv``:

```bash
uv sync --no-dev   # runtime only
```

Development installation with ``uv``:

```bash
uv sync            # runtime + dev
```

For installation without ``uv`` (for instance if using Conda), we can fall back to ``pip``:

```bash
pip install .
```