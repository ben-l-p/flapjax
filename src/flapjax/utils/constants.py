from __future__ import annotations

from typing import Literal

# minimum value of ||h_omega||^2 before reverting to small angle approximation for SE(3)/SO(3)
# functions with singularities (exp, log, tangent and inverse tangent)
SMALL_ANG_THRESH = 1e-6

# summation order for computing se(3)/so(3) operators from infinite series
BASE_SUMMATION_ORDER = 15

# length of horseshoe wake, in meters
HORSESHOE_LENGTH = 1e3

# small number to avoid division by zero in Biot-Savart kernel with epsilon regularisation
EPSILON = 1e-7

# cutoff radius for Biot-Savart kernel, in meters
R_CUTOFF = 1e-4

# default order of integration for Lobatto and Legendre quadrature rules
BASE_LOBATTO_ORDER: Literal[3, 4, 5] = 3
BASE_LEGENDRE_ORDER: Literal[1, 2, 3] = 3
