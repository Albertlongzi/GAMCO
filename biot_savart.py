"""
Biot-Savart Law implementation for calculating magnetic fields from coils.
This is a Python conversion of the MATLAB BiotSavart100PointsParallel.m function.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor

def biot_savart_single_coil(coil_data, grid_data, current=1.0):
    """
    Calculate the magnetic field (Bz component) for a single coil using the Biot-Savart law.
    
    Parameters:
    -----------
    coil_data : tuple
        Tuple containing (x_coords, y_coords, z_coords) of the coil points
    grid_data : tuple
        Tuple containing (x_grid, y_grid, z_grid) of the calculation grid
    current : float
        Current flowing through the coil in Amperes
    
    Returns:
    --------
    numpy.ndarray
        3D array of Bz field values
    """
    x1f, y1f, z1f = coil_data
    x, y, z = grid_data
    
    nums = len(x1f)
    bz_local = np.zeros(x.shape, dtype=np.float64)
    
    for k in range(nums):
        # Get next point (wrap around to first point if at the end)
        m = 0 if k == nums - 1 else k + 1
        
        # Calculate current element vector
        dlx = x1f[m] - x1f[k]
        dly = y1f[m] - y1f[k]
        dlz = z1f[m] - z1f[k]
        
        # Calculate position vector from current element to field point
        rx = x - (x1f[k] + x1f[m]) / 2
        ry = y - (y1f[k] + y1f[m]) / 2
        rz = z - (z1f[k] + z1f[m]) / 2
        
        # Calculate distance and its cube
        rsq = np.sqrt(rx**2 + ry**2 + rz**2)
        r3 = rsq**3
        
        # Calculate Bz component using Biot-Savart law
        # Original MATLAB: BzcLocal = BzcLocal + 1e-7 * I * (dlx * rz - dlz * rx) ./ r3;
        bz_local += 1e-7 * current * (dlx * rz - dlz * rx) / r3
    
    return bz_local

def biot_savart_parallel(coil_coords, n_coils, x_grid, y_grid, z_grid, current=1.0):
    """
    Calculate the magnetic field (Bz component) for multiple coils using the Biot-Savart law.
    This function can use parallel processing for faster computation.
    
    Parameters:
    -----------
    coil_coords : numpy.ndarray
        Array of shape (3, num_points, n_coils) containing the coordinates of each coil
    n_coils : int
        Number of coils
    x_grid, y_grid, z_grid : numpy.ndarray
        3D arrays representing the grid points where the field will be calculated
    current : float
        Current flowing through the coils in Amperes
    
    Returns:
    --------
    numpy.ndarray
        4D array of Bz field values with shape (grid_shape, n_coils)
    """
    # Initialize output array
    bz = np.zeros((*x_grid.shape, n_coils))
    
    # Process each coil
    for i in range(n_coils):
        x1f = coil_coords[0, :, i]
        y1f = coil_coords[1, :, i]
        z1f = coil_coords[2, :, i]
        
        # Calculate field for this coil
        bz[:, :, :, i] = biot_savart_single_coil(
            (x1f, y1f, z1f), 
            (x_grid, y_grid, z_grid), 
            current
        )
    
    return bz

def biot_savart_parallel_multiprocessing(coil_coords, n_coils, x_grid, y_grid, z_grid, current=1.0, max_workers=None):
    """
    Calculate the magnetic field using multiple processes for better performance on multi-core systems.
    
    Parameters are the same as biot_savart_parallel, with the addition of:
    
    max_workers : int or None
        Maximum number of worker processes to use. If None, uses the number of CPU cores.
    """
    # Initialize output array
    bz = np.zeros((*x_grid.shape, n_coils))
    
    # Prepare arguments for parallel execution
    args_list = []
    for i in range(n_coils):
        x1f = coil_coords[0, :, i]
        y1f = coil_coords[1, :, i]
        z1f = coil_coords[2, :, i]
        args_list.append(((x1f, y1f, z1f), (x_grid, y_grid, z_grid), current))
    
    # Execute in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda args: biot_savart_single_coil(*args), args_list))
    
    # Collect results
    for i, result in enumerate(results):
        bz[:, :, :, i] = result
    
    return bz
