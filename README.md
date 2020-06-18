## ENRGY
Physical-based distributed glacier melt model.

Computes ice and snow ablation based on heat amount available for melt.

Temperature of melting surface is assumed constant and equaling 0°C

Heat available for melt is calculated from glacier surface heat budget equation, which incorporates:
* shortwave radiation flux
* longwave radiation flux
* turbulent heat fluxes (assuming stable atmospheric conditions)

Not included yet:
* ground heat flux
* sensible heat flux of rain

Input data are meteorological observations from AWS at the glacier surface, albedo map and DEM.
