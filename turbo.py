"""
Bulk aerodynamic technique for estimating heat exchange in turbulent flow
See references:
1) Munro, D.S. 1989. Surface roughness and bulk heat transfer on a
glacier: comparison with eddy correlation. J. Glaciol., 35(121),
343–348.
2) Munro, D.S. 1990. Comparison of melt energy computations and
ablatometer measurements on melting ice and snow. Arct. Alp.
Res., 22(2), 153–162.
3) Beljaars, A. and A. Holtslag. 1991. Flux parameterization over land
surface for atmospheric models. J. Appl. Meteorol., 30(3),
327–341.
4) Hock, R., & Holmgren, B. (2005). A distributed surface energy-balance model
for complex topography and its application to Storglaciären, Sweden.
Journal of Glaciology, 51(172), 25-36. doi:10.3189/172756505781829566
5) Wheler, B., & Flowers, G. (2011). Glacier subsurface heat-flux characterizations for energy-balance modelling in the Donjek Range,
southwest Yukon, Canada. Journal of Glaciology, 57(201), 121-133. doi:10.3189/002214311795306709

"""
import numpy as np


def calc_turbulent_fluxes(z, uz, Tz, P, rel_humidity, max_iter=5, verbose=False):
	sensible_flux, monin_obukhov_lentgh = _calc_sensible_iteratively(z, uz, Tz, P, max_iter=max_iter, verbose=verbose)
	latent_flux = _calc_latent(z, uz, Tz, P, rel_humidity, L=monin_obukhov_lentgh)
	return sensible_flux, latent_flux


def _get_dry_air_density(t_air, p_air):
	specific_gas_constant = 287.058  # J kg-1 K-1
	return p_air/(specific_gas_constant * t_air)


def _calc_sensible_iteratively(z, uz, Tz, P, max_iter=5, verbose=False):
	# calculation of L requires knowledge of both Qh and the friction velocity u_aster:
	# we need to make an initial guess of L
	# assuming that z/L = 0 by passing l=None into the following functions:
	u_aster_init = _calc_friction_velocity(uz, z, L=None)
	Qh_init = _calc_sensible(z, uz, Tz, P, L=None)
	L_init = _calc_monin_obukhov_length(Tz, P, u_aster_init, Qh_init)

	if verbose:
		print("Initial guess:")
		print("u*=%.3f m/s" % u_aster_init)
		print("Qh=%.1f W/m^2" % Qh_init)
		print("Monin-Obukhov length is %.1f m" % L_init)

	u_aster = u_aster_init
	Qh = Qh_init
	L = L_init

	for i in range(0, max_iter):
		u_aster = _calc_friction_velocity(uz, z, L=L)
		Qh = _calc_sensible(z, uz, Tz, P, L=L)
		L = _calc_monin_obukhov_length(Tz, P, u_aster, Qh)
		if verbose:
			print("***************************")
			print("Iteration %d" % (i+1))
			print("u*=%.3f m/s" % u_aster)
			print("Qh=%.1f W/m^2" % Qh)
			print("Monin-Obukhov length is %.1f m" % L)

	return Qh, L


def _calc_monin_obukhov_length(Tz, P, u_aster, Qh):
	"""
	Computes Monin-Obukhov stability length [m]
	:param Tz: absolute air temperature at the height "z" above the surface [K]
	:param P: air pressure at the height "z" [Pa]
	:param u_aster: friction velocity [m s-1]
	:param Qh: sensible heat flux [W m-2]
	:return:
	"""
	k = 0.4  # dimensionless, von Karman constant
	g = 9.81  # meters per s^2, acceleration due to gravity
	Cp = 1010  # J kg-1 K-1,  the specific heat capacity of air
	rho = _get_dry_air_density(Tz, P)  # kg m-3, air density
	num = rho * Cp * u_aster ** 3 * Tz
	denum = k * g * Qh
	return num / denum


def _calc_sensible(z, uz, Tz, P, L=None):
	"""
	Computes sensible heat flux [W m-2]
	:return:
	"""
	Ts = 0 + 273.15  # K, temperature of melting surface
	Cp = 1010  # J kg-1 K-1,  the specific heat capacity of air
	rho = _get_dry_air_density(Tz, P)  # kg m-3, air density
	CH = _calc_turb_exchange_coef(z, L=L)
	# print("CH coefficient is %s" % CH)
	return CH * Cp * rho * uz * (Tz - Ts)


def _calc_latent(z, uz, Tz, P, rel_humidity, L=None):
	"""
	Computes latent heat flux [W m-2]
	:param z:
	:param uz:
	:param Tz:
	:param P:
	:param rel_humidity:
	:param L:
	:return:
	"""
	es = 611  # Pa, water vapour pressure at the ice surface
	e_max = _calc_e_max(Tz, P)  # Pa, partial water vapor pressure for saturated air
	ez = e_max * rel_humidity  # Pa, partial vapour pressure at the height of measurements "z"
	Lv = 2.514 * 10 ** 6  # J kg-1, latent heat of vaporization (for positive flux)
	Ls = 2.849 * 10 ** 6  # J kg-1, latent heat of sublimation (for negative flux)
	rho = _get_dry_air_density(Tz, P)  # kg m-3, air density
	CE = _calc_turb_exchange_coef(z, L=L)
	# print("CE coefficient is %s" % CE)
	return CE * Lv * rho * uz * 0.622 / P * (ez - es)


def _calc_turb_exchange_coef(z, L=None):
	"""
	Computes CH or CE coefficients
	:param z: height of measurements above the surface [m]
	:param L: Monin-Obukhov stability length [m]
	:return:
	"""
	k = 0.4  # dimensionless, von Karman constant
	zm = 0.001  # meters, roughness length for momentum
	z_h_or_e = zm / 100  # meters, roughness length for heat or water vapour
	num = k ** 2
	if L is not None:
		minus_psi_m = _calc_minus_psi_m(z, L)
		minus_psi_h_or_e = _calc_minus_psi_h_or_e(z, L)
		denum = (np.log(z / zm) + minus_psi_m * (z / L)) * (np.log(z / z_h_or_e) + minus_psi_h_or_e * (z / L))
	else:
		denum = np.log(z / zm) * np.log(z / z_h_or_e)
	return num / denum


def _calc_friction_velocity(uz, z, L=None):
	k = 0.4  # dimensionless, von Karman constant
	zm = 0.01  # meters, roughness length for momentum
	num = k * uz
	if L is not None:
		minus_psi_m = _calc_minus_psi_m(z, L)
		# denum = np.log(z / zm) + minus_psi_m * (z / L)
		denum = np.log(z / zm) + minus_psi_m
	else:
		denum = np.log(z / zm)
	return num / denum


def _calc_minus_psi_m(z, L):
	"""
	Computes stability constant Psi-M
	:return:
	"""
	a = 0.7
	b = 0.75
	c = 5
	d = 0.35
	return a * z / L + b * (z / L - c / d) * np.exp(-d * z / L) + b * c / d


def _calc_minus_psi_h_or_e(z, L):
	"""
	Computes stability constant Psi-H or Psi-E
	:return:
	"""
	a = 0.7
	b = 0.75
	c = 5
	d = 0.35
	return (1 + 2 * a * z / 3 * L) ** 1.5 + b * (z / L - c / d) * np.exp(-d * z / L) + b * c / d - 1


def _calc_e_max(t_air, air_pressure):
	"""
	Computes partial water vapour pressure of saturated (100%-moist) air [Pa, not hPa and not kPa!]
	:param t_air: in Kelvin
	:param air_pressure: in Pascals
	:return:
	"""
	t_air = t_air - 273.15
	air_pressure = air_pressure / 100
	ew_t = 611.2 * np.exp((17.62 * t_air) / (243.12 + t_air))
	f_p = 1.0016 + 3.15 * 10**-6 * air_pressure - 0.074 / air_pressure
	return f_p * ew_t


if __name__ == "__main__":
	z = 1.6  # m
	uz = 2.5  # m/s
	Tz = 3 + 273.15  # K
	P = 99000  # Pascals
	rel_humidity = 0.85
	############################
	Qh, L = _calc_sensible_iteratively(z, uz, Tz, P)
	latent = _calc_latent(z, uz, Tz, P, rel_humidity, L)
	print("")
	print("Latent flux is %.1f W/m^2" % latent)
