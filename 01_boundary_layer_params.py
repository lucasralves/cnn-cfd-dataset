"""
    Calcualte the boundary layer mesh parameters
"""
from math import ceil, log
from typing import List

def boundary_layer_mesh(velocity: float, density: float, viscosity: float, reynolds: float, exp_ratio: float) -> List[float]:
    """Returns the first element height, boundary layer height and de number of elements."""
    
    # Boundary layer height
    delta = 1.5 * max([max(5.0 / (reynolds ** 0.5), 0.37 / (reynolds ** 0.2))])

    # Shear stress
    tau_w = max([0.5 * density * velocity * velocity * 0.074 / (reynolds ** (1 / 5))])

    # First element
    y1 = min([(viscosity / density) * ((density / tau_w) ** 0.5)])

    # Number of layers
    n = ceil(log(1 + delta * (exp_ratio - 1) / y1, exp_ratio))

    return [delta, n, y1]

if __name__ == '__main__':
    
    # Environment
    velocity = 12.0
    density = 1.225
    viscosity = 1.46 * density
    reynolds = velocity * density / viscosity

    # Mesh
    ns = 700
    nt = 10
    nf = None
    epsilon1 = 1.02
    epsilon2 = 1.02
    epsilon3 = 1.1
    delta = None
    radius = 25
    ext_grid_size = 2

    delta, nf, y1 = boundary_layer_mesh(velocity, density, viscosity, reynolds, epsilon3)

    print('delta: {}; nf: {}; y1: {}'.format(delta, nf, y1))