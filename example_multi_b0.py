"""
Example script demonstrating how to use the MSK GA OPT code with multiple B0 fields.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from biot_savart import biot_savart_parallel
from coil_geometry import generate_coordinates, generate_mesh_grid
from optimization import prepare_fields_for_solve, solve_dc_currents

def objective_function_multi_b0(params, affine_matrix, position_data, b0_data_list, bz_data, mask_data_list, f_central, weights=None):
    """
    Objective function for optimization with multiple B0 fields.
    
    Parameters:
    -----------
    params : numpy.ndarray
        Array of parameters to optimize (y_positions, z_positions, radius_indices)
    affine_matrix : numpy.ndarray
        Affine transformation matrix
    position_data : numpy.ndarray
        Position data for the region of interest
    b0_data_list : list of numpy.ndarray
        List of B0 field data arrays
    bz_data : numpy.ndarray
        Bz field data for existing shim coils
    mask_data_list : list of numpy.ndarray
        List of mask data arrays for each B0 field
    f_central : float
        Central frequency
    weights : list of float, optional
        Weights for each B0 field in the cost function
    
    Returns:
    --------
    float
        Weighted cost value (standard deviation of B0 fields after shimming)
    """
    # Set default weights if not provided
    if weights is None:
        weights = [1.0] * len(b0_data_list)
    
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    
    # Number of coils
    n_coils = len(params) // 3
    
    # Extract parameters
    y_positions = params[:n_coils]
    z_positions = params[n_coils:2*n_coils]
    radius_indices = np.round(params[2*n_coils:]).astype(int)
    
    # Define radius choices and get actual radii
    radius_choices = np.array([0.035, 0.055])  # in meters
    coil_radii = radius_choices[radius_indices]
    
    # Cylinder parameters
    R = 0.280  # cylinder radius (m)
    height = 0.200  # cylinder height (m)
    boundary_bottom = -0.097  # cylinder bottom z-coordinate
    boundary_left = -0.260  # cylinder left x-coordinate
    
    # Calculate center coordinates
    center_x = boundary_left + np.sqrt(R**2 - (height/2)**2)
    center_z = boundary_bottom + height / 2
    
    # Initialize coil positions
    coil_positions = np.zeros((n_coils, 3))
    
    # Calculate x-coordinates based on y and z positions
    for i in range(n_coils):
        y = y_positions[i] / 1000.0  # convert to meters
        z = z_positions[i] / 1000.0  # convert to meters
        
        # Solve cylinder equation for x
        x = center_x - np.sqrt(R**2 - (z - center_z)**2)
        
        # Store coordinates
        coil_positions[i] = [x, y, z]
    
    # Calculate tilt angles
    tilt_angles = np.zeros(n_coils)
    for i in range(n_coils):
        if coil_positions[i, 2] > 0:
            tilt_angles[i] = np.rad2deg(np.arccos((coil_positions[i, 0] - center_x) / R)) + 90
        else:
            tilt_angles[i] = np.rad2deg(-np.arccos((coil_positions[i, 0] - center_x) / R)) + 90
    
    # Generate coil coordinates
    coil_coords = generate_coordinates(n_coils, coil_radii, coil_positions, tilt_angles)
    
    # Generate mesh grid
    resolution = [affine_matrix[0, 0], affine_matrix[1, 1], affine_matrix[2, 2]]
    x_grid, y_grid, z_grid = generate_mesh_grid(position_data, resolution)
    
    # Calculate magnetic field
    bz_mapped = biot_savart_parallel(coil_coords, n_coils, x_grid, y_grid, z_grid)
    
    # Extract SH coil fields
    bz_sh = bz_data[:, :, :, :8]  # Assuming first 8 channels are SH coils
    
    # Initialize cost
    total_cost = 0.0
    
    # Process each B0 field
    for i, (b0_data, mask_data, weight) in enumerate(zip(b0_data_list, mask_data_list, weights)):
        # Create a bounding box mask
        rows, cols, pages = np.where(mask_data > 0)
        x_min, x_max = np.min(rows), np.max(rows)
        y_min, y_max = np.min(cols), np.max(cols)
        z_min, z_max = np.min(pages), np.max(pages)
        
        new_mask = np.zeros_like(mask_data, dtype=bool)
        new_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = True
        
        # Prepare coil fields
        bz_ncoil = np.zeros((*bz_mapped.shape[:3], n_coils))
        for j in range(n_coils):
            bz_replacement = np.copy(bz_data[:, :, :, 0])
            bz_replacement[new_mask] = bz_mapped[:, :, :, j][new_mask]
            bz_ncoil[:, :, :, j] = bz_replacement
        
        # Combine all fields
        bz_combined = np.concatenate((bz_ncoil, bz_sh), axis=3)
        
        # Prepare fields for solving
        b0f, bzf = prepare_fields_for_solve(bz_combined, b0_data, mask_data)
        
        # Solve for optimal currents
        dc = solve_dc_currents(b0f, bzf, 50, f_central, '0-2SH+7 channel')
        
        # Calculate shimmed field
        shimfield = np.zeros_like(bz_combined)
        for c in range(bz_combined.shape[3]):
            shimfield[:, :, :, c] = -dc[c] * bz_combined[:, :, :, c]
        
        shimf = np.sum(shimfield, axis=3) * f_central
        b0freq_sim = b0_data * f_central + shimf
        
        # Calculate statistics within mask
        mask_indices = mask_data > 0
        b0freq_sim_masked = b0freq_sim[mask_indices]
        
        # Calculate standard deviation as cost
        std_b0 = np.nanstd(b0freq_sim_masked)
        
        # Add weighted cost
        total_cost += weight * std_b0
    
    return total_cost

def run_multi_b0_optimization(affine_matrix, position_data, b0_data_list, bz_data, mask_data_list, f_central, 
                             n_coils=7, pop_size=50, max_iter=20, weights=None):
    """
    Run the optimization process with multiple B0 fields.
    
    Parameters:
    -----------
    affine_matrix : numpy.ndarray
        Affine transformation matrix
    position_data : numpy.ndarray
        Position data for the region of interest
    b0_data_list : list of numpy.ndarray
        List of B0 field data arrays
    bz_data : numpy.ndarray
        Bz field data for existing shim coils
    mask_data_list : list of numpy.ndarray
        List of mask data arrays for each B0 field
    f_central : float
        Central frequency
    n_coils : int
        Number of coils
    pop_size : int
        Population size for the optimization algorithm
    max_iter : int
        Maximum number of iterations
    weights : list of float, optional
        Weights for each B0 field in the cost function
    
    Returns:
    --------
    tuple
        Tuple containing (optimal_params, optimal_cost)
    """
    # Define bounds for optimization
    bounds = []
    # Y-position bounds: -100 to 100 mm
    bounds.extend([(-100, 100) for _ in range(n_coils)])
    # Z-position bounds: -25 to 25 mm
    bounds.extend([(-25, 25) for _ in range(n_coils)])
    # Radius index bounds: 0 to 1 (will be rounded to integers)
    bounds.extend([(0, 1) for _ in range(n_coils)])
    
    # Wrap objective function
    wrapped_objective = lambda x: objective_function_multi_b0(
        x, affine_matrix, position_data, b0_data_list, bz_data, mask_data_list, f_central, weights
    )
    
    # Run differential evolution (similar to genetic algorithm)
    result = differential_evolution(
        wrapped_objective,
        bounds,
        popsize=pop_size,
        maxiter=max_iter,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        workers=-1,  # Use all available cores
        disp=True,
        polish=True
    )
    
    return result.x, result.fun

def main():
    """
    Main function to demonstrate the optimization process with multiple B0 fields.
    Note: This requires actual data files to run.
    """
    print("MSK GA Optimization with Multiple B0 Fields")
    print("This is a demonstration script that requires actual data files.")
    print("To use this code with your data, you need to:")
    print("1. Load your affine matrix, position data, multiple B0 data, Bz data, and masks")
    print("2. Call run_multi_b0_optimization with your data")
    print("3. Visualize the results")
    
    # Example usage (commented out as it requires actual data)
    """
    # Load data
    affine_matrix = np.load('affine_matrix.npy')
    position_data = np.load('position_data.npy')
    
    # Load multiple B0 fields and masks
    b0_data_1 = np.load('b0_data_1.npy')
    b0_data_2 = np.load('b0_data_2.npy')
    b0_data_3 = np.load('b0_data_3.npy')
    
    mask_data_1 = np.load('mask_data_1.npy')
    mask_data_2 = np.load('mask_data_2.npy')
    mask_data_3 = np.load('mask_data_3.npy')
    
    bz_data = np.load('bz_data.npy')
    f_central = 123.2e6  # Example central frequency
    
    # Define weights for each B0 field (optional)
    weights = [0.4, 0.3, 0.3]  # Prioritize the first B0 field
    
    # Run optimization
    optimal_params, optimal_cost = run_multi_b0_optimization(
        affine_matrix, position_data, 
        [b0_data_1, b0_data_2, b0_data_3], 
        bz_data, 
        [mask_data_1, mask_data_2, mask_data_3], 
        f_central,
        weights=weights
    )
    
    # Extract optimized parameters
    n_coils = len(optimal_params) // 3
    y_positions = optimal_params[:n_coils]
    z_positions = optimal_params[n_coils:2*n_coils]
    radius_indices = np.round(optimal_params[2*n_coils:]).astype(int)
    
    # Print results
    print(f"Optimal cost (weighted std): {optimal_cost}")
    print("Optimized parameters:")
    for i in range(n_coils):
        print(f"Coil {i+1}: Y={y_positions[i]:.1f} mm, Z={z_positions[i]:.1f} mm, "
              f"Radius={35 if radius_indices[i]==0 else 55} mm")
    """

if __name__ == "__main__":
    main()
