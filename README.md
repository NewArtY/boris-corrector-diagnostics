# Reproduction bundle

Everything here is under `code/`. Start at `code/README.md`; the single-command
driver is `code/repro.py`.

| what | where |
| :-- | :-- |
| experiment scripts, and the committed JSON/NPZ they produce | `code/experiments/` |
| the integrators and field configurations under test | `code/fields/`, `code/models/` |
| trained defect-corrector checkpoints | `code/checkpoints/` |
| the four figure scripts | `code/figures/` |
| one-command reproduction | `code/repro.py` |
| how to cite | `CITATION.cff` (repository root) |
| Zenodo deposit metadata | `.zenodo.json` (repository root) |

## Two things worth knowing before you run it

**`code/` is not a Python package, and must not become one.** There is no
`code/__init__.py`, deliberately: an `__init__.py` here would make this
directory shadow the standard library module named `code`, and `import pdb`
would then fail — which breaks debugging in PyCharm and VS Code, and breaks
`import torch` with it. If a tool offers to add one, decline.

**The result this bundle reproduces is in part negative.** The learned defect
corrector is outperformed by a fourth-order volume-preserving splitting at the
same step and at a fraction of the cost, and its trajectory advantage inverts
beyond roughly one hundred gyro-orbits. That is not a defect of the bundle. Every
number supporting the statement is in the committed outputs, and the scripts that
produced them are here.

## Provenance

The first version of this work, titled "Separating physical particle heating from
numerical drift in non-stationary magnetic fields", was withdrawn. Its LaTeX
source and figures are not part of this repository and are not in its history:
its figures are not the figures of the present paper and its numbers are not the
numbers of the present paper. They exist only in the authors' local working copy,
which is where they belong.

Licence: MIT, see `code/LICENSE`.
