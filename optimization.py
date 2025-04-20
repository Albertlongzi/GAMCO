"""
Optimization functions for coil current calculation.
This is a Python conversion of the MATLAB SolveDC_coilsetup_7channel_msk.m function.
"""

import numpy as np
from scipy.optimize import lsq_linear

def solve_dc_currents(b0f, bzf, dc_limit, f_central, shimming_setup):
    """
    Solve for optimal DC currents in the coil setup.

    Parameters:
    -----------
    b0f : numpy.ndarray
        Flattened B0 field values
    bzf : numpy.ndarray
        Matrix of Bz field values for each coil/channel
    dc_limit : float
        Current limit for the coils
    f_central : float
        Central frequency
    shimming_setup : str
        Shimming setup type ('0-2SH_only', '0-2SH+7 channel', '7 channel_only', etc.)

    Returns:
    --------
    numpy.ndarray
        Array of optimal DC currents for each coil/channel
    """
    # Get number of channels
    channel_num = bzf.shape[1]

    # Add a 0th order term
    bzf_extended = np.hstack((bzf, np.zeros((bzf.shape[0], 1))))

    # Set current limits
    lb = -np.ones(channel_num + 1) * dc_limit
    ub = np.ones(channel_num + 1) * dc_limit

    # Set specific limits for different channels
    # Channels N-7 to N-5
    lb[channel_num-7:channel_num-5+1] = -5000
    ub[channel_num-7:channel_num-5+1] = 5000

    # Channel N-4
    lb[channel_num-4] = -1839
    ub[channel_num-4] = 1839

    # Channels N-3 to N-2
    lb[channel_num-3:channel_num-2+1] = -791
    ub[channel_num-3:channel_num-2+1] = 791

    # Channels N-1 to N
    lb[channel_num-1:channel_num+1] = -615
    ub[channel_num-1:channel_num+1] = 615

    # 0th order term
    lb[channel_num] = -3000
    ub[channel_num] = 3000

    # Configure based on shimming setup
    if shimming_setup == '0-2SH_only':
        # No UNIC. SH + 1/f_central
        bzf_extended[:, 0:7] = 0
        bzf_extended[:, 15] = 1/f_central

    elif shimming_setup == '0-2SH+7 channel':
        # Keep all currents, and add one more
        bzf_extended[:, channel_num] = 1/f_central

    elif shimming_setup == '7 channel_only':
        # Only use 7 channel
        bzf_extended[:, channel_num-7:channel_num] = 0
        bzf_extended[:, channel_num] = 1/f_central

    elif shimming_setup == 'BOT UNIC+ 0-2SH+7 channel':
        bzf_extended[:, 57] = 1/f_central
        bzf_extended[:, 36:57] = 0

    elif shimming_setup == 'BOT UNIC + 0-2SH':
        bzf_extended[:, 57] = 1/f_central
        bzf_extended[:, 36:57] = 0
        bzf_extended[:, 0:7] = 0

    elif shimming_setup == '0-1SH+7 channel':
        bzf_extended[:, 57] = 1/f_central
        bzf_extended[:, 15:57] = 0
        bzf_extended[:, 10:15] = 0

    elif shimming_setup == 'BOT UNIC+TOP UNIC+ 0-2SH':
        bzf_extended[:, 57] = 1/f_central
        bzf_extended[:, 0:7] = 0

    else:
        print('No such coil setup, went with SH+Both_plate')

    # Initial values
    x0 = np.zeros(channel_num + 1)

    # Solve the linear least squares problem
    result = lsq_linear(bzf_extended, b0f, bounds=(lb, ub), method='trf',
                        lsmr_tol='auto', verbose=0)

    # Reshape the result
    dc = result.x

    return dc

def prepare_fields_for_solve(bz, b0, mask, is_slice=False, shim_slice=None):
    """
    Prepare B0 and Bz fields for solving.
    This is a direct Python implementation of the MATLAB Prep_B_forSolve script.

    Parameters:
    -----------
    bz : numpy.ndarray
        4D array of Bz field values for each coil/channel
    b0 : numpy.ndarray
        3D array of B0 field values
    mask : numpy.ndarray
        3D binary mask indicating the region of interest
    is_slice : bool, optional
        Flag indicating whether to use a specific slice for shimming
    shim_slice : numpy.ndarray, optional
        Indices of slices to use for shimming if is_slice is True

    Returns:
    --------
    tuple
        Tuple containing (b0f, bzf) - flattened arrays ready for optimization
    """
    # Handle slice selection
    if is_slice and shim_slice is not None:
        slab = shim_slice
    else:
        slab = np.arange(b0.shape[2])  # whole volume

    # Handle NaN values
    mask = mask.copy()
    mask[np.isnan(b0)] = 0
    bz_clean = bz.copy()
    bz_clean[np.isnan(bz_clean)] = 0

    # Get dimensions
    nx, ny, nz, nc = bz_clean.shape

    # Initialize Bzf matrix
    bzf = np.zeros((np.sum(mask > 0), nc))

    # Fill Bzf matrix with masked Bz values for each channel
    for i in range(nc):
        bz_temp = bz_clean[:, :, :, i]
        bzf[:, i] = bz_temp[mask > 0]

    # Extract B0 values within the mask
    b0f = b0[mask > 0]

    return b0f, bzf
