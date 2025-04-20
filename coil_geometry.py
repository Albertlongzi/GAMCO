"""
Functions for generating coil geometries.
This is a Python conversion of the MATLAB generate_coordinates.m function.
"""

import numpy as np

def generate_coordinates(n_coils, coil_radius, coil_position, tilt_angle):
    """
    Generate coordinates for circular coils.
    
    Parameters:
    -----------
    n_coils : int
        Number of coils
    coil_radius : numpy.ndarray
        Array of coil radii (in meters)
    coil_position : numpy.ndarray
        Array of coil positions, shape (n_coils, 3) for [x, y, z] coordinates (in meters)
    tilt_angle : numpy.ndarray
        Array of tilt angles (in degrees)
    
    Returns:
    --------
    numpy.ndarray
        Array of coil coordinates with shape (3, num_points, n_coils)
    """
    num_points = 100
    coil_coords = np.zeros((3, num_points, n_coils))
    
    # Generate theta values for the circle
    theta = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]  # Remove the last point to avoid duplication
    
    # Generate coordinates for each coil
    for i in range(n_coils):
        # Get parameters for this coil
        a = coil_radius[i]  # semi-major axis
        b = coil_radius[i]  # semi-minor axis (for circular coils, a = b)
        
        # Initial coordinates (planar circle)
        coil_coords[2, :, i] = coil_position[i, 2] * np.ones(num_points)
        coil_coords[1, :, i] = coil_position[i, 1] + b * np.cos(theta)
        coil_coords[0, :, i] = coil_position[i, 0] + a * np.sin(theta)
        
        # Translate to origin for rotation
        coil_coords[:, :, i] = coil_coords[:, :, i] - coil_position[i, :].reshape(3, 1)
        
        # Convert tilt angle from degrees to radians
        tilt_angle_rad = np.deg2rad(tilt_angle[i])
        
        # Rotation matrix around Y-axis
        Ry = np.array([
            [0, np.cos(tilt_angle_rad), -np.sin(tilt_angle_rad)],
            [1, 0, 0],
            [0, np.sin(tilt_angle_rad), np.cos(tilt_angle_rad)]
        ])
        
        # Apply rotation
        coil_coords[:, :, i] = Ry @ coil_coords[:, :, i]
        
        # Translate back to original position
        coil_coords[:, :, i] = coil_coords[:, :, i] + coil_position[i, :].reshape(3, 1)
    
    return coil_coords

def generate_mesh_grid(position, resolution):
    """
    Generate a 3D mesh grid for field calculations.
    This is a Python conversion of the MATLAB generateMeshGrid.m function.
    
    Parameters:
    -----------
    position : numpy.ndarray
        Array of positions, shape (n, 3) for [x, y, z] coordinates
    resolution : list or numpy.ndarray
        Resolution in each dimension [res_x, res_y, res_z]
    
    Returns:
    --------
    tuple
        Tuple containing (x_grid, y_grid, z_grid) arrays
    """
    # Extract min and max values for each dimension
    x_min = np.min(position[:, 0])
    x_max = np.max(position[:, 0])
    y_min = np.min(position[:, 1])
    y_max = np.max(position[:, 1])
    z_min = np.min(position[:, 2])
    z_max = np.max(position[:, 2])
    
    # Create point array
    point = np.array([
        [x_min, y_min, z_min],
        [x_max, y_max, z_max]
    ])
    
    # Generate mesh grid
    y, x, z = np.meshgrid(
        np.arange(point[0, 1], point[1, 1] + resolution[1], resolution[1]),
        np.arange(point[0, 0], point[1, 0] + resolution[0], resolution[0]),
        np.arange(point[0, 2], point[1, 2] + resolution[2], resolution[2])
    )
    
    return x, y, z
