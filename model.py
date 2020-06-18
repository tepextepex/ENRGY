import gdal
import numpy as np
import matplotlib.pyplot as plt
from turbo import calc_turbulent_fluxes
from turbo import _calc_e_max
from turbo import _calc_sensible, _calc_latent


class Energy:
	def __init__(self, base_dem_path, glacier_outlines_path, albedo_path, potential_incoming_radiation_path):

		self.__init_constants()

		self.outlines_path = glacier_outlines_path

		print("Loading base DEM...")
		self.base_dem_array = self._load_raster(base_dem_path, self.outlines_path)

		print("Loading albedo map...")
		self.albedo = self._load_raster(albedo_path, self.outlines_path, remove_outliers=True)
		debug_imshow(self.albedo, "Albedo grid")

		print("Loading insolation map...")
		self.potential_incoming_radiation = self._load_raster(potential_incoming_radiation_path, self.outlines_path)
		debug_imshow(self.potential_incoming_radiation, "Total potential incoming radiation kW*h month-1")

	def __init_constants(self):
		self.CONST = {
			"ice_density": 916.7,  # kg m-3
			"latent_heat_of_fusion": 3.34 * 10**5,  # J kg-1
			"g": 9.81
		}

	def set_aws_data(self, t_air_aws, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, z, elev_aws):
		self.t_air_aws = t_air_aws
		self.wind_speed = wind_speed
		self.rel_humidity = rel_humidity
		self.air_pressure = air_pressure
		self.cloudiness = cloudiness
		self.incoming_shortwave = incoming_shortwave
		self.z = z
		self.elev_aws = elev_aws

		self.Tz = self.t_air_aws + 273.15
		self.P = self.air_pressure * 100  # Pascals from hPa

	def run(self, days):
		# heat available for melt:
		Qm = self.calc_heat_influx(self.t_air_aws, self.wind_speed, self.rel_humidity, self.air_pressure, self.cloudiness, self.incoming_shortwave, self.z, self.elev_aws)
		debug_imshow(Qm, title="Heat available for melt")

		# amount of ice which was melt:
		ice_density = self.CONST["ice_density"]
		ice_melt = self.calc_ice_melt(Qm, ice_density, days)
		debug_imshow(ice_melt, title="Ice melt, m w.e.")
		return ice_melt

	def calc_heat_influx(self, t_air_aws, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, z, elev_aws):
		t_surf = 0  # we assume that surface of melting ice is 0 degree Celsius

		# we need four numpy arrays with meteorological values:
		t_air_array = self.interpolate_air_t(t_air_aws, elev_aws)
		debug_imshow(t_air_array, title="Air temperature, degree Celsius")

		wind_speed_array = self.fill_array_with_one_value(wind_speed)  # wind speed is non-distributed and assumed constant across the surface
		debug_imshow(wind_speed_array, title="Wind speed")

		air_pressure_array = self.interpolate_pressure(air_pressure, elev_aws)
		debug_imshow(air_pressure_array, title="Air pressure")

		e_aws = _calc_e_max(t_air_aws + 273.15, air_pressure * 100) * rel_humidity  # this is partial pressure of water vapour at the AWS
		# print("Emax at AWS is %.3f" % _calc_e_max(t_air_aws + 273.15, air_pressure * 100))
		e_array = self.interpolate_e(e_aws, elev_aws)
		# print("E at AWS is %.3f" % e_aws)
		debug_imshow(e_array, title="Partial pressure of water vapour")
		e_max_array = _calc_e_max(t_air_array + 273.15, air_pressure_array * 100)
		debug_imshow(e_max_array, title="Partial pressure of saturated air (e_max)")
		rel_humidity_array = e_array / e_max_array
		debug_imshow(rel_humidity_array, title="Relative humidity")

		# TURBULENT HEAT FLUXES
		# at the AWS:
		sensible_flux, latent_flux, monin_obukhov_length = calc_turbulent_fluxes(self.z, self.wind_speed, self.Tz, self.P, self.rel_humidity)

		# at the whole glacier surface:
		sensible_flux_array, latent_flux_array, monin_obukhov_length = calc_turbulent_fluxes(self.z, wind_speed_array, t_air_array + 273.15, air_pressure_array * 100, rel_humidity_array, L=monin_obukhov_length)
		debug_imshow(sensible_flux_array, title="Sensible heat flux")
		debug_imshow(latent_flux_array, title="Latent heat flux")

		# LONGWAVE RADIATION FLUX
		rl = self.calc_longwave(t_air_array, t_surf, cloudiness)
		debug_imshow(rl, title="Longwave")

		# SHORTWAVE RADIATION FLUX
		self.incoming_shortwave = self.J_to_W(self.kWh_to_J(self.potential_incoming_radiation))
		self.incoming_shortwave = self.incoming_shortwave * 0.5 / 30  # 30 days
		debug_imshow(self.incoming_shortwave, title="Real incoming solar radiation")

		rs = self.calc_shortwave(self.incoming_shortwave)
		debug_imshow(rs, title="Incoming shortwave * (1 - albedo)")

		return rl + rs + sensible_flux_array + latent_flux_array

	def calc_shortwave(self, incoming_shortwave):
		return incoming_shortwave * (1 - self.albedo)

	def calc_ice_melt(self, ice_heat_influx, ice_density, days):
		"""

		:param ice_heat_influx:
		:param ice_density: assumed ice density kg per cibic meter
		:param days: number of days
		:return: thickness of melt ice layer in meters w.e.
		"""
		latent_heat_of_fusion = self.CONST["latent_heat_of_fusion"]
		return (ice_heat_influx * 86400) / (ice_density * latent_heat_of_fusion) * days

	@staticmethod
	def calc_longwave(t_air, t_surf, cloudiness):
		lwu = 0.98 * 5.669 * 10 ** -8 * to_kelvin(t_surf) ** 4
		lwd = (0.765 + 0.22 * cloudiness ** 3) * 5.669 * 10 ** -8 * to_kelvin(t_air) ** 4
		return lwd - lwu

	@staticmethod
	def _load_raster(raster_path, crop_path, remove_negatives=False, remove_outliers=False):
		ds = gdal.Open(raster_path)
		crop_ds = gdal.Warp("", ds, dstSRS="+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs", format="VRT", cutlineDSName=crop_path, cropToCutline=True, outputType=gdal.GDT_Float32, xRes=10, yRes=10)
		band = crop_ds.GetRasterBand(1)
		nodata = band.GetNoDataValue()
		array = band.ReadAsArray()
		array[array == nodata] = np.nan
		if remove_negatives:
			array[array < 0] = np.nan  # makes sense for albedo which couldn't be negative
		if remove_outliers:
			array[array > 1] = np.nan
		print("Raster size is %dx%d" % array.shape)
		return array

	@staticmethod
	def show_me(array):
		plt.imshow(array)
		plt.colorbar()
		plt.show()
		plt.clf()

	@staticmethod
	def kWh_to_J(insol):
		"""
		Converts amount of energy in [kW*h / day] into [J/day]
		:param insol:
		:return:
		"""
		return insol * 3.6 * 10 ** 6

	@staticmethod
	def J_to_W(insol):
		"""
		Converts amount of energy in [J/day] into the flux in [W/s]
		:param insol:
		:return:
		"""
		return insol / 86400

	def interpolate_e(self, e, elev):
		"""
		Interpolates partial pressure of water vapour (e)
		from measured at the AWS at the given elevation
		:param e:
		:param elev:
		:return:
		"""
		delta_elev = self.base_dem_array - elev
		return e * 10**(-delta_elev / 6300)

	def interpolate_air_t(self, t_air, elev, v_gradient=None):
		if v_gradient is None:
			v_gradient = -0.006  # 6 degrees Celsius or Kelvin per 1 m
		return self.interpolate_on_dem(t_air, elev, v_gradient)

	def interpolate_pressure(self, pressure, elev, v_gradient=None):
		"""

		:param pressure: in hPa
		:param elev:
		:param v_gradient:
		:return:
		"""
		if v_gradient is None:
			v_gradient = -0.1145  # Pa per 1 m
		return self.interpolate_on_dem(pressure, elev, v_gradient)

	def interpolate_on_dem(self, value, elev, v_gradient):
		"""
		Interpolates a single measurement of something, which depends on elevation from a dem, to a numpy array
		:param value: measured value
		:param elev: measurement elevation (above sea or ellipsoid level, NOT above surface)
		:param v_gradient: vertical gradient
		:return: array
		"""
		return (self.base_dem_array - elev) * v_gradient + value

	def fill_array_with_one_value(self, value):
		array = np.zeros_like(self.base_dem_array, dtype=np.float32)
		array[~np.isnan(self.base_dem_array)] = value
		array[np.isnan(self.base_dem_array)] = np.nan
		return array


def to_kelvin(t_celsius):
	return t_celsius + 273.15


def debug_imshow(array, title="Test", show=False):
	try:
		plt.imshow(array)
		plt.title(title)
		print("Mean %s is %.3f:" % (title, np.nanmean(array)))
		plt.colorbar()
		plt.savefig("/home/tepex/PycharmProjects/energy/png/%s.png" % title)
		if show:
			plt.show()
		plt.clf()
	except Exception as e:
		print(e)
