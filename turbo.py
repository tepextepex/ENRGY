import numpy as np
"""
Bulk aerodynamic technique for estimating heat exchange in turbulent flow
See references:
Munro, D.S. 1989. Surface roughness and bulk heat transfer on a
glacier: comparison with eddy correlation. J. Glaciol., 35(121),
343–348.
Munro, D.S. 1990. Comparison of melt energy computations and
ablatometer measurements on melting ice and snow. Arct. Alp.
Res., 22(2), 153–162.
Beljaars, A. and A. Holtslag. 1991. Flux parameterization over land
surface for atmospheric models. J. Appl. Meteorol., 30(3),
327–341.
Hock, R., & Holmgren, B. (2005). A distributed surface energy-balance model
for complex topography and its application to Storglaciären, Sweden.
Journal of Glaciology, 51(172), 25-36. doi:10.3189/172756505781829566
Wheler, B., & Flowers, G. (2011). Glacier subsurface heat-flux characterizations for energy-balance modelling in the Donjek Range,
southwest Yukon, Canada. Journal of Glaciology, 57(201), 121-133. doi:10.3189/002214311795306709

"""


def get_dry_air_density(t_air, p_air):
	specific_gas_constant = 287.058  # J kg-1 K-1
	return p_air/(specific_gas_constant * t_air)


def calc_sensible_iteratively(z, uz, Tz, P, max_iter=5):
	# calculation of L requires knowledge of both Qh and the friction velocity u_aster:
	# we need to make an initial guess of L
	# assuming that z/L = 0 by passing l=None into the following functions:
	print("Initial guess:")
	u_aster_init = calc_friction_velocity(uz, z, L=None)
	print("u*=%.3f m/s" % u_aster_init)
	Qh_init = calc_sensible(z, uz, Tz, P, L=None)
	print("Qh=%.1f W/m^2" % Qh_init)
	L_init = calc_monin_obukhov_length(Tz, P, u_aster_init, Qh_init)
	print("Monin-Obukhov length is %.1f m" % L_init)

	u_aster = u_aster_init
	Qh = Qh_init
	L = L_init

	for i in range(0, max_iter):
		print("***************************")
		print("Iteration %d" % (i+1))
		u_aster = calc_friction_velocity(uz, z, L=L)
		print("u*=%.3f m/s" % u_aster)
		Qh = calc_sensible(z, uz, Tz, P, L=L)
		print("Qh=%.1f W/m^2" % Qh)
		L = calc_monin_obukhov_length(Tz, P, u_aster, Qh)
		print("Monin-Obukhov length is %.1f m" % L)


def calc_monin_obukhov_length(Tz, P, u_aster, Qh):
	k = 0.4  # dimensionless, von Karman constant
	g = 9.81  # meters per s^2, acceleration due to gravity
	Cp = 1010  # J kg-1 K-1,  the specific heat capacity of air
	rho = get_dry_air_density(Tz, P)  # kg m-3, air density
	num = rho * Cp * u_aster ** 3 * Tz
	denum = k * g * Qh
	return num / denum


def calc_sensible(z, uz, Tz, P, L=None):
	"""
	Computes Qh
	:return:
	"""
	Ts = 0 + 273.15  # K, temperature of melting surface
	Cp = 1010  # J kg-1 K-1,  the specific heat capacity of air
	rho = get_dry_air_density(Tz, P)  # kg m-3, air density
	CH = calc_turb_exchange_coef(z, L=L)
	# print("CH coefficient is %s" % CH)
	return CH * Cp * rho * uz * (Tz - Ts)


def calc_latent(z, uz, Tz, P, rel_humidity, L=None):
	es = 611  # Pa, water vapour pressure at the ice surface
	e_max = calc_e_max(Tz, P)  # Pa, partial water vapor pressure for saturated air
	ez = e_max * rel_humidity  # Pa, partial vapour pressure at the height of measurements "z"
	Lv = 2.514 * 10 ** 6  # K kg-1, latent heat of vaporization
	rho = get_dry_air_density(Tz, P)  # kg m-3, air density
	CE = calc_turb_exchange_coef(z, L=L)
	# print("CE coefficient is %s" % CE)
	return CE * Lv * rho * uz * 0.622 / P * (ez - es)


def calc_turb_exchange_coef(z, L=None):
	"""
	Computes CH or CE
	:param z:
	:param zm:
	:param z_h_or_e:
	:param minus_psi_m:
	:param minus_psi_h_or_e:
	:param L:
	:return:
	"""
	k = 0.4  # dimensionless, von Karman constant
	zm = 0.001  # meters, roughness length for momentum
	z_h_or_e = zm / 100  # meters, roughness length for heat or water vapour
	num = k ** 2
	if L is not None:
		minus_psi_m = calc_minus_psi_m(z, L)
		minus_psi_h_or_e = calc_minus_psi_h_or_e(z, L)
		denum = (np.log(z / zm) + minus_psi_m * (z / L)) * (np.log(z / z_h_or_e) + minus_psi_h_or_e * (z / L))
	else:
		denum = np.log(z / zm) * np.log(z / z_h_or_e)
	return num / denum


def calc_friction_velocity(uz, z, L=None):
	k = 0.4  # dimensionless, von Karman constant
	zm = 0.01  # meters, roughness length for momentum
	num = k * uz
	if L is not None:
		minus_psi_m = calc_minus_psi_m(z, L)
		# denum = np.log(z / zm) + minus_psi_m * (z / L)
		denum = np.log(z / zm) + minus_psi_m
	else:
		denum = np.log(z / zm)
	return num / denum


def calc_minus_psi_m(z, L):
	"""
	Computes stability constant
	:return:
	"""
	a = 0.7
	b = 0.75
	c = 5
	d = 0.35
	return a * z / L + b * (z / L - c / d) * np.exp(-d * z / L) + b * c / d


def calc_minus_psi_h_or_e(z, L):
	"""
	Computes stability constant
	:return:
	"""
	a = 0.7
	b = 0.75
	c = 5
	d = 0.35
	return (1 + 2 * a * z / 3 * L) ** 1.5 + b * (z / L - c / d) * np.exp(-d * z / L) + b * c / d - 1


def calc_e_max(t_air, air_pressure):
	"""
	Computes partial water vapour pressure of saturated air IN PASCALS
	:param t_air: in Kelvin
	:param air_pressure: in Pascals
	:return:
	"""
	t_air = t_air - 273.15
	air_pressure = air_pressure / 100
	ew_t = 611.2 * np.exp((17.62 * t_air) / (243.12 + t_air))
	f_p = 1.0016 + 3.15 * 10**-6 * air_pressure - 0.074 / air_pressure
	return f_p * ew_t


calc_sensible_iteratively(1.6, 2.5, 3 + 273.15, 99000)
latent = calc_latent(1.6, 2.5, 3 + 273.15, 99000, 0.85, 62.4)
print("")
print("Latent flux is %.1f W/m^2" % latent)
