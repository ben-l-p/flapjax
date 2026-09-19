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

This framework is designed to be modular, and allows for very flexible workflows. It makes extensive use of
object-oriented principles with a very "pythonic" interface. It makes full use of Python type hinting to best inform
users of the expected input and output types for each function.

<div class="grid cards" markdown>

- :material-school: **[Inputs and outputs](inputs_outputs/index.md)**
  How to construct an aeroelastic model, and what the solver returns.

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

## Core principles

### JAX

flapjax is written in JAX, with the advantages of this listed above. However, this also comes with some limitations and
quirks. It is advised that users have an understanding of key JAX principles, including JIT compilation, vectorisation,
and automatic differentiation [JAX documentation](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html). Some
of the key points to note are:

- Flow control logic. JAX does not allow for python flow control logic (e.g. ``if`` statements,) to be used in
  JIT-compiled functions. It instead provides its own set of functional control flow operators, such as ``jax.lax.cond``
  and ``jax.lax.while_loop``. However, there are more restrictive. For example, code cannot exit early; in a time domain
  solve, if a solution diverges, the code will continue to run until the end of the time domain, giving NaN results.
- We cannot run python routines or use python debugging within JIT-compiled functions. Therefore, we cannot plot during
  analysis, and so all plotting routines for Paraview can only be used once the full solution is complete.
- Solvers may take a while to start iterating, and look as if they are stuck. JAX traces code to create a computational
  graph of the function to be JIT-compiled. In exteme cases, this can take a few minutes to compile.
- Internal numerics use the `jax.numpy` library, which mostly mirrors the regular `numpy` library. This requires us to
  use jax-type arrays for our inputs/outputs, which are not compatible with regular numpy arrays in many contexts:

  ```python
  # numpy arrays - do not use, may cause unexpected behaviour
  import numpy as np
  
  array_np = np.array([1, 2, 3])
  
  # jax arrays, which should be used
  import jax.numpy as jnp
  
  array_jnp = jnp.array([1, 2, 3])
  ```

### Inputs and Outputs

To keep the codebase as simple, we do not use input files to define our models. Instead, we define the model, which
creates the model in the form of a Python object. A common workflow is as follows

1) Create the model using a Python function written by the user (examples are provided in the tutorials) This object
   contains all of the information that defines the model (reference coordinates, element stiffness, aerodynamic
   geometry, velocity, etc). Creating this model does not perform any analysis.
2) Use the model object to perform analysis, which are member functions of the model - many solvers are available. There
   is no limit to the number of analyses that can be performed on a single model object, as the model object is
   immutable. These solvers will return a solution object, which contains the results of the analysis, for example
   displacements, aerodynamic forces, etc. We may often use the solution object as an input to another analysis.
   This approach can allow for efficient workflows; for example, reusing a trimmed static solution as the initial
   condition for a number of dynamic analysis.
3) Extract or plot the desired information from the solution object. It is important to note that by default, flapjax
   will not plot or save any information to disk. There are two ways of extracting information from the solution object:
    - Plotting solutions to Paraview using the ``solution.plot()`` functions.
    - Manually extracting information from the solution object. Details on what is available in the solution object can
      be found in the reference documentation.

```angular2html
# (1) - create the model with some user defined function
aircraft = create_model(u_inf, rho, ...)

# (2) - perform an analysis; here we show a dynamic aeroelastic solution, initialised at the static equilbrium
static_solution = aircraft.static_solve(...)  # static aeroelastic analysis
dynamic_solution = aircraft.dynamic_solve(init_cond=static_solution, ...)  # dynamic aeroelastic analysis

# (3) - extract information from the solution object
node_coords = dynamic_solution.structure.x  # example - get the structural node coordinates from the dynamic solution
jnp.save("node_coords.npy", node_coords)  # save the node coordinates to disk for later use
dynamic_solution.plot("plot_dir")  # plot the dynamic solution to the specified directory for visualisation in Paraview
```

### Computational efficiency

This code is designed to be computationally efficient, and can be run on CPU or GPU hardware. It is important to note
that whilst the time to solve a problem is almost always faster than equivelant codes (e.g., SHARPy), for small
problems, the time to compile the code may be longer than the time to solve the problem. This relative cost can be
reduced by reusing functions, for example, using the same JIT compilation to run multiple gusts.

Whilst the code has been tested to run on GPU, it does not always outperform CPU hardware. The relative performance is
dependent on both the problem size and the solver being used.

The compile and run time performance of flapjax has also been observed to vary significantly depending on the hardware
used.
