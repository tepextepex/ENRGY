"""
Multilayer Subsurface Model (of a glacier).

Simulates surface temperature changes according to supra-glacial heat balance.
Yields in-glacier heat flux and a flux, needed to warp the surface up to zero degrees Celsius.

Based on equations from:
(Lisette) Klok, E., & Oerlemans, J. (2002). Model study of the spatial distribution of the energy
and mass balance of Morteratschgletscher, Switzerland. Journal of Glaciology, 48(163), 505-518.
doi:10.3189/172756502781831133

"""
from dataclasses import dataclass, field
from typing import List
from parameter_classes import CONST


@dataclass
class Layer:
    depth: float  # meters
    temp: float  # degree Celsius


@dataclass
class Msm:
    thickness_list: List[float]
    temp_list: List[float]
    layers: List[Layer] = field(init=False)
    c: float = field(init=False)  # specific heat capacity of ice
    k: float = field(init=False)  # thermal diffusivity
    density: float = field(init=False)

    def __post_init__(self):
        # initializing constants:
        self.c = CONST["specific_heat_capacity_ice"]
        self.k = CONST["thermal_diffusivity_ice"]
        self.density = CONST["ice_density"]
        # initializing layers:
        self.layers = []
        if len(self.thickness_list) and len(self.temp_list):
            for i in range(0, len(self.thickness_list)):
                self.layers.append(Layer(self.thickness_list[i], self.temp_list[i]))

    def tick(self, timestep, atmo_heat_flux):
        print("Msm timestep is %s seconds" % timestep)
        grad_t = (self.layers[1].temp - self.layers[0].temp) / (self.layers[1].depth - self.layers[0].depth)
        glacier_heat_flux = self.k * grad_t * self.c * self.density
        print("Glacier heat flux is %.1f W/m^2" % glacier_heat_flux)
        delta_temp_surf = self.k * grad_t / self.layers[1].depth + atmo_heat_flux / (self.c * self.density * self.layers[1].depth)
        print("Temperature change rate is %.6f/s or %.1f/step" % (delta_temp_surf, delta_temp_surf * timestep))
        full_balance = atmo_heat_flux + glacier_heat_flux
        print("Full heat balance is %.1f W/m^2" % full_balance)
        # flux, needed to warm the surface up to 0 degrees:
        zero_flux = -self.layers[0].temp * self.c * self.density * self.layers[1].depth / timestep
        print("Zero-flux is %.1f W/m^2" % zero_flux)
        melt_flux = 0 if full_balance <= zero_flux else full_balance - zero_flux
        print("Melt flux is %.1f W/m^2" % melt_flux)
        return melt_flux


if __name__ == "__main__":
    m = Msm([0, 0.22, 2.78], [-0.05, -0.15, -2.5])
    print(m)
    m.tick(3600, 100)
