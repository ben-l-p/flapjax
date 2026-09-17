# flapjax-full

`flapjax-full` bundles [FLAPJAX](https://pypi.org/project/flapjax/) together with its test suite and tutorial notebooks,
for users who want the complete codebase installable via `pip` without cloning the repository.

Full documentation, including tutorials, API reference and theory, is available at
[ben-l-p.github.io/flapjax](https://ben-l-p.github.io/flapjax/).

## Installation

Two PyPI distributions are provided:

| Package        | Contents                                  |
|----------------|-------------------------------------------|
| `flapjax`      | Library only                              |
| `flapjax-full` | Library + test suite + tutorial notebooks |

Install one distribution - they provide the same `flapjax` import path.

```bash
pip install flapjax        # minimal
pip install flapjax-full   # includes tests and tutorial notebooks
```

## Running the tests

```bash
pytest --pyargs flapjax
```
