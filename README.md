## <img alt="Logo" src="https://drive.google.com/uc?export=view&id=18f7qEgNjjDJxzW6cLkSXEg7VFcjywmgL" height="32">ENRGY
Physical-based distributed glacier melt model.

Computes ice and snow ablation based on heat amount available for melt.

##### Assumptions
* temperature of melting surface is assumed to be constant and equaling 0°C.
Hence, the water vapour partial pressure at the surface always equals 611 Pa;
* meltwater is removed instantaneously - neither percolation nor refreezing is simulated.

##### Heat balance equation
Heat available for melt is calculated from glacier surface heat budget equation, which incorporates:
* shortwave radiation flux;
* longwave radiation flux;
* turbulent heat fluxes (assuming stable atmospheric conditions).

##### Model constraints
* ground (in-glacier) heat flux is not modelled;
* penetration of shorwave radiation below surface is not modelled;
* sensible heat flux of rain is not modelled;
* NOTE: bulk aerodynamic technique, used for turbulent heat fluxes, is not the most precise one.

Input data are meteorological observations from AWS at the glacier surface, DEM and albedo map.

##### Prerequisutes
The model script uses:
* gdal
* numpy
* matplotlib
