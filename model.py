import gdal
import numpy as np
import matplotlib.pyplot as plt


class Energy:
	def __init__(self, base_dem_path, glacier_outlines_path, albedo_path, insolation_path):

		print("Loading base DEM...")
		self.base_dem_array = self._load_raster(base_dem_path, glacier_outlines_path)

		print("Loading albedo map...")
		self.constant_albedo = self._load_raster(albedo_path, glacier_outlines_path)

		print("Loading insolation map...")
		self.constant_insolation = self._load_raster(insolation_path, glacier_outlines_path)

	def calc_heat_influx(self, t_air_aws, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, albedo, z, elev_aws):
		t_surf = 0  # we assume that surface of melting ice is 0 degree Celsius
		t_air = self.interpolate_air_t(t_air_aws, elev_aws)
		hs = self.calc_sensible_heat_transfer(t_air, t_surf, wind_speed)
		hl = self.calc_latent_heat_transfer(t_air, rel_humidity, wind_speed, air_pressure, z)
		rl = self.calc_longwave(t_air, t_surf, cloudiness)

		debug_imshow(hs, title="Sensible")
		debug_imshow(hl, title="Latent")
		debug_imshow(rl, title="Longwave")

		return hs + hl + rl + incoming_shortwave * (1 - albedo)

	@staticmethod
	def calc_ice_melt(ice_heat_influx, ice_density, days):
		"""

		:param ice_heat_influx:
		:param ice_density: assumed ice density kg per cibic meter
		:param days: number of days
		:return: thickness of melt ice layer in meters w.e.
		"""
		return (ice_heat_influx * 86400) / (ice_density * 3.33 * 10**5) * days

	@staticmethod
	def calc_longwave(t_air, t_surf, cloudiness):
		lwu = 0.98 * 5.669 * 10 ** -8 * to_kelvin(t_surf) ** 4
		lwd = (0.765 + 0.22 * cloudiness ** 3) * 5.669 * 10 ** -8 * to_kelvin(t_air) ** 4
		# return lwu - lwd
		return lwd - lwu

	@staticmethod
	def calc_shortwave(k, potential_incoming_solar_radiation):
		return potential_incoming_solar_radiation * k

	@staticmethod
	def calc_sensible_heat_transfer(t_air, t_surf, wind_speed):
		return 1.293 * 1005 * 0.001 * wind_speed * (t_surf - t_air)

	def calc_latent_heat_transfer(self, t_air, rel_humidity, wind_speed, air_pressure, z):
		q_air_0 = self.calc_q(rel_humidity, t_air, air_pressure, z=z)  # yes, z=z, that is not a typo
		print("q_air_0 is %.3f" % np.nanmean(q_air_0))

		q_air_z = self.calc_q(rel_humidity, t_air, air_pressure, z=0)  # yes, z=0 is here
		print("q_air_z is %.3f" % np.nanmean(q_air_z))

		return 1.293 * 2260 * 0.01 * wind_speed * (q_air_0 - q_air_z)

	def calc_q(self, rel_humidity, t_air, air_pressure, z=0):
		"""
		Computes specific water content of the air at the certain level above the surface
		:param rel_humidity: measured air relative humidity
		:param t_air: measured air temperature in Celsius
		:param air_pressure: in hPa
		:param z: measurements height above the surface in meters
		:return:
		"""
		# variable "e" is the partial water vapour pressure measured in hPa
		e_max = self.calc_e_max(t_air, air_pressure)  # partial water vapor pressure for saturated air
		print("max partial water vapour pressure is %.3f hPa" % np.nanmean(e_max))
		ez = (rel_humidity * e_max)  # e at the height of measurements
		# print("%.3f hPa" % np.nanmean(ez))
		e0 = ez / (10 ** (-z / 6300))  # e at the needed level
		print("%.3f hPa" % np.nanmean(e0))
		p = (18.015 * e0) / (8.31 * t_air)
		return (623 * e0) / (p - 0.377 * e0)

	@staticmethod
	def calc_e_max(t_air, air_pressure):
		"""
		Computes partial water vapour pressure of saturated air
		:param t_air: in Celsius
		:param air_pressure: in hPa
		:return:
		"""
		ew_t = np.empty_like(t_air, dtype=np.float32)
		ew_t = 6.112 * np.exp((17.62 * t_air) / (243.12 + t_air))
		f_p = 1.0016 + 3.15 * 10**-6 * air_pressure - 0.074 / air_pressure
		return f_p * ew_t

	@staticmethod
	def _load_raster(raster_path, crop_path):
		ds = gdal.Open(raster_path)
		crop_ds = gdal.Warp("", ds, dstSRS="+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs", format="VRT", cutlineDSName=crop_path, outputType=gdal.GDT_Float32, xRes=10, yRes=10)
		band = crop_ds.GetRasterBand(1)
		nodata = band.GetNoDataValue()
		array = band.ReadAsArray()
		array[array == nodata] = np.nan
		return array

	@staticmethod
	def show_me(array):
		plt.imshow(array)
		plt.colorbar()
		plt.show()
		plt.clf()

	def interpolate_air_t(self, t_air, elev):
		"""

		:param t_air: measured air temperature at the weather station
		:param elev: elevation of the weather station
		:return: numpy array with air temperature grid
		"""
		# temp_array = np.empty_like(self.base_dem_array, dtype=np.float32)
		return (self.base_dem_array - elev) * -6/1000 + t_air


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
