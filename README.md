## ENRGY
Physical-based distributed glacier melt model.

Computes ice and snow ablation based on heat amount available for melt.

Heat available for melt is calculated from glacier surface heat budget equation, which incorporates:
* shortwave radiation flux
* longwave radiation flux
* turbulent heat fluxes (sensible and latent)

Input data are meteorological observations from AWS at the glacier surface, albedo map and DEM.
