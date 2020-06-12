import gdal
import numpy as np
import matplotlib.pyplot as plt
from math import exp


class Energy:
	def __init__(self, base_dem_path, glacier_outlines_path, albedo_path, insolation_path):

		print("Loading base DEM...")
		self.base_dem_array = self._load_raster(base_dem_path, glacier_outlines_path)

		print("Loading albedo map...")
		self.constant_albedo = self._load_raster(albedo_path, glacier_outlines_path)

		print("Loading insolation map...")
		self.constant_insolation = self._load_raster(insolation_path, glacier_outlines_path)

	def calc_heat_influx(self, t_air, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, albedo, z):
		t_surf = 0  # we assume that surface of melting ice is 0 degree Celsius
		hs = self.turb_heat_transfer(t_air, t_surf, wind_speed)
		hl = self.latent_heat(t_air, rel_humidity, wind_speed, air_pressure, z)
		rl = self.calc_longwave(t_air, t_surf, cloudiness)
		return hs + hl + rl + incoming_shortwave * (1 - albedo)

	@staticmethod
	def calc_ice_melt(ice_heat_influx, ice_density, days):
		"""

		:param ice_heat_influx:
		:param ice_density: assumed ice density kg per cibic meter
		:param days: number of days
		:return: thickness of melt ice layer in meters w.e.
		"""
		return (ice_heat_influx * 86400) / (ice_density * 3.33 * 10e5) * days

	@staticmethod
	def calc_longwave(t_air, t_surf, cloudiness):
		lwu = 0.98 * 5.669 * 10e-8 * to_kelvin(t_surf) ** 4
		lwd = (0.765 + 0.22 * cloudiness ** 3) * 5.669 * 10e-8 * to_kelvin(t_air) ** 4
		return lwu - lwd

	@staticmethod
	def calc_shortwave(k, potential_incoming_solar_radiation):
		return potential_incoming_solar_radiation * k

	@staticmethod
	def turb_heat_transfer(t_air, t_surf, wind_speed):
		return 1.293 * 1005 * 0.001 * wind_speed * (t_surf - t_air)

	def latent_heat(self, t_air, rel_humidity, wind_speed, air_pressure, z):
		q_air_0 = self.calc_q(rel_humidity, t_air, air_pressure, z=z)  # yes, z=z, that is not a typo
		q_air_z = self.calc_q(rel_humidity, t_air, air_pressure, z=0)  # yes, z=0 is here
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
		# variable "e" is the partial water vapour pressure
		e_max = self.calc_e_max(t_air, air_pressure)  # partial water vapor pressure for saturated air
		ez = (rel_humidity * e_max) / 100  # e at the height of measurements
		e0 = ez / (10 ** (-z / 6300))  # e at the needed level
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
		ew_t = 6.112 * exp((17.62 * t_air) / (243.12 + t_air))
		f_p = 1.0016 + 3.15 * 10e-6 * air_pressure - 0.074 / air_pressure
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
		plt.show()
		plt.clf()


def to_kelvin(t_celsius):
	return t_celsius + 273.15
