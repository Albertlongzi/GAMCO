"""
MSK GA OPT Code - Python Implementation

This package contains a Python implementation of the MSK GA OPT code,
which is used for optimizing the positions and radii of electromagnetic coils
for magnetic field shimming in MRI applications.
"""

from .biot_savart import biot_savart_parallel, biot_savart_parallel_multiprocessing
from .coil_geometry import generate_coordinates, generate_mesh_grid
from .optimization import prepare_fields_for_solve, solve_dc_currents
from .msk_ga_optimization import objective_function, run_optimization, visualize_coils

__all__ = [
    'biot_savart_parallel',
    'biot_savart_parallel_multiprocessing',
    'generate_coordinates',
    'generate_mesh_grid',
    'prepare_fields_for_solve',
    'solve_dc_currents',
    'objective_function',
    'run_optimization',
    'visualize_coils'
]
