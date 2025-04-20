# MSK GA OPT Code - Python Implementation

This repository contains a Python implementation of the MSK GA OPT code, which is used for optimizing the positions and radii of electromagnetic coils for magnetic field shimming in MRI applications.

## Overview

The MSK GA OPT code uses a genetic algorithm (or similar optimization techniques) to find optimal coil configurations that minimize the standard deviation of the B0 magnetic field within a region of interest (ROI). This helps improve MRI image quality by reducing magnetic field inhomogeneities.

A key feature of this implementation is the ability to consider multiple B0 fields (up to three) as the target area for coil arrangement optimization. This allows for more comprehensive optimization across different regions or imaging scenarios.

## Files

- `biot_savart.py`: Implementation of the Biot-Savart law for calculating magnetic fields from coils
- `coil_geometry.py`: Functions for generating coil geometries and mesh grids
- `optimization.py`: Functions for solving optimal coil currents
- `msk_ga_optimization.py`: Main script demonstrating the optimization process
- `example_multi_b0.py`: Example script demonstrating optimization with multiple B0 fields

## Usage

To use this code with your own data:

1. Prepare your input data:
   - Affine transformation matrix
   - Position data for the region of interest
   - B0 field data (single or multiple fields)
   - Bz field data for existing shim coils
   - Mask data for the region of interest (corresponding to each B0 field)
   - Central frequency value

2. Run the optimization:
   ```python
   from msk_ga_optimization import run_optimization

   # Load your data
   # ...

   # Run optimization
   optimal_params, optimal_cost = run_optimization(
       affine_matrix, position_data, b0_data, bz_data, mask_data, f_central
   )
   ```

3. Extract and use the optimized parameters:
   ```python
   # Extract optimized parameters
   n_coils = len(optimal_params) // 3
   y_positions = optimal_params[:n_coils]
   z_positions = optimal_params[n_coils:2*n_coils]
   radius_indices = np.round(optimal_params[2*n_coils:]).astype(int)

   # Convert radius indices to actual radii
   radius_choices = [35, 55]  # in mm
   coil_radii = [radius_choices[idx] for idx in radius_indices]

   # Print results
   print(f"Optimal cost (std): {optimal_cost}")
   print("Optimized parameters:")
   for i in range(n_coils):
       print(f"Coil {i+1}: Y={y_positions[i]:.1f} mm, Z={z_positions[i]:.1f} mm, "
             f"Radius={coil_radii[i]} mm")
   ```

### Multiple B0 Field Optimization

This implementation supports optimization across multiple B0 fields, which is useful for designing coil arrays that work well across different anatomical regions or imaging scenarios:

```python
from example_multi_b0 import run_multi_b0_optimization

# Load multiple B0 fields and masks
b0_data_list = [b0_data_1, b0_data_2, b0_data_3]
mask_data_list = [mask_data_1, mask_data_2, mask_data_3]

# Define weights for each B0 field (optional)
weights = [0.4, 0.3, 0.3]  # Prioritize the first B0 field

# Run optimization
optimal_params, optimal_cost = run_multi_b0_optimization(
    affine_matrix, position_data,
    b0_data_list, bz_data, mask_data_list,
    f_central, weights=weights
)
```

The weights parameter allows you to prioritize certain B0 fields over others in the optimization process.

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib

## Notes and Limitations

1. **Missing Prep_B_forSolve Script**: The original MATLAB code references a script called `Prep_B_forSolve` which is not included in the provided codebase. We've implemented a likely version of this functionality in the `prepare_fields_for_solve` function in `optimization.py`.

2. **Data Format Differences**: The Python implementation expects data in a slightly different format than the MATLAB code. Make sure to adapt your data accordingly.

3. **Optimization Algorithm**: The original MATLAB code uses MATLAB's Genetic Algorithm, while this Python implementation uses SciPy's Differential Evolution, which is similar but may produce slightly different results.

4. **Parallel Processing**: The Biot-Savart calculation supports parallel processing for better performance on multi-core systems.

5. **Visualization**: Basic visualization functions are included, but you may need to adapt them for your specific needs.

## Extending the Code

To extend this code for your specific application:

1. Modify the objective function in `msk_ga_optimization.py` to include additional constraints or optimization goals.

2. Adjust the coil geometry parameters in `coil_geometry.py` to match your hardware setup.

3. Implement additional shimming strategies in `optimization.py` if needed.

## References

This code is based on the MATLAB MSK GA OPT code, which was developed for optimizing coil positions for magnetic field shimming in MRI applications.

For more information on the methodology and applications, please refer to:

Long Z, et al. "Optimized local coil array design for musculoskeletal MRI." Magnetic Resonance in Medicine. 2024. DOI: 10.1002/mrm.30474

https://onlinelibrary.wiley.com/doi/10.1002/mrm.30474

## License

This code is provided for research and educational purposes only. Please respect the original authors' work if you use or modify this code.
