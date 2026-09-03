"""Electromagnetic field configurations used throughout the study.

Field hierarchy (see Fig. 1 / fig1_field_hierarchy.py):
  uniform_field.py     - homogeneous B, analytic cyclotron reference
  dipole_field.py       - dipole B ~ (r0/|r|)^3, spatial inhomogeneity
  stochastic_field.py   - stochastically phase-modulated B, OOD generalization test
  radial_field.py   (B1) - quadratic radial gradient
  wave_field.py     (B2) - weak spatiotemporal wave
  tilted_field.py   (B3) - mixed x/z tilted static field
  decaying_field.py (B4) - time-decaying B(t) = B0 exp(-t/tau), key physics case
"""
from .uniform_field import UniformField
from .dipole_field import DipoleField
from .stochastic_field import StochasticField
from .radial_field import RadialField
from .wave_field import WaveField
from .tilted_field import TiltedField
from .decaying_field import DecayingField

__all__ = [
    "UniformField", "DipoleField", "StochasticField",
    "RadialField", "WaveField", "TiltedField", "DecayingField",
]
