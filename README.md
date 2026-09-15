# FLAPJAX — FLexible Adjoint-enabled Panel code in JAX

*nonlinear · differentiable · adjoint-enabled*

![Tests](https://github.com/ben-l-p/flapjax/actions/workflows/python_package.yml/badge.svg)
[![cov](https://ben-l-p.github.io/flapjax/badges/coverage.svg)](https://github.com/ben-l-p/flapjax/actions)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://ben-l-p.github.io/flapjax/)
[![PyPI](https://img.shields.io/pypi/v/flapjax)](https://pypi.org/project/flapjax/)
![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/ben-l-p/flapjax/main/pyproject.toml)

FLAPJAX is a differentiable nonlinear aeroelastic analysis framework which couples a nonlinear structural model with
unsteady vortex lattice method (UVLM) aerodynamics. This allows for a range of structural, aerodynamic and coupled
analyses, with gradients available using the adjoint method. The full codebase is implemented in JAX, which allows for
efficient automatic differentiation, GPU acceleration and multi-case parallelisation.

Full documentation, including tutorials, API reference and theory, is available at
[ben-l-p.github.io/flapjax](https://ben-l-p.github.io/flapjax/).

## Installation

### Github

For the latest version from git, clone the repository and install with pip:

```bash
git clone https://github.com/ben-l-p/flapjax/
cd flapjax
pip install -e .  # uses the -e flag to install in editable mode
```

### PyPI

Two PyPI distributions are provided:

| Package        | Contents                                  |
|----------------|-------------------------------------------|
| `flapjax`      | Library only                              |
| `flapjax-full` | Library + test suite + tutorial notebooks |

Install one distribution — they provide the same `flapjax` import path.

```bash
pip install flapjax        # minimal
pip install flapjax-full   # includes tests and tutorial notebooks
```

## Testing

An extensive suite of over 300 tests is included to verify the correctness of the code. This checks the numerics, and
takes approximately 40 minutes to run on an M2 MacBook Air. Tests can be run with pytest, either against a
`flapjax-full`
install (`pytest --pyargs flapjax`) or from a repository clone (`uv run pytest`).



