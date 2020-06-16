import gdal
import numpy as np
import matplotlib.pyplot as plt


class Energy:
	def __init__(self, base_dem_path, glacier_outlines_path, albedo_path, insolation_path):

		self.outlines_path = glacier_outlines_path

		print("Loading base DEM...")
		self.base_dem_array = self._load_raster(base_dem_path, self.outlines_path)

		print("Loading albedo map...")
		self.constant_albedo = self._load_raster(albedo_path, self.outlines_path, remove_outliers=True)
		debug_imshow(self.constant_albedo, "Albedo grid (not used)")

		print("Loading insolation map...")
		self.constant_insolation = self._load_raster(insolation_path, self.outlines_path)
		debug_imshow(self.constant_insolation, "Total potential incoming radiation (not used)")

	def run(self, t_air_aws, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, albedo, z, elev_aws, ice_density, days):
		# heat available for melt:
		Qm = self.calc_heat_influx(t_air_aws, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, albedo, z, elev_aws)
		debug_imshow(Qm, title="Heat available for melt")
		# amount of ice which was melt:
		ice_melt = self.calc_ice_melt(Qm, ice_density, days)
		debug_imshow(ice_melt, title="Ice melt, m w.e.")
		return ice_melt

	def calc_heat_influx(self, t_air_aws, wind_speed, rel_humidity, air_pressure, cloudiness, incoming_shortwave, albedo, z, elev_aws):
		t_surf = 0  # we assume that surface of melting ice is 0 degree Celsius
		t_air = self.interpolate_air_t(t_air_aws, elev_aws)

		rl = self.calc_longwave(t_air, t_surf, cloudiness)
		debug_imshow(rl, title="Longwave")

		return rl + incoming_shortwave * (1 - albedo)

	@staticmethod
	def calc_ice_melt(ice_heat_influx, ice_density, days):
		"""

		:param ice_heat_influx:
		:param ice_density: assumed ice density kg per cibic meter
		:param days: number of days
		:return: thickness of melt ice layer in meters w.e.
		"""
		return (ice_heat_influx * 86400) / (ice_density * 3.33 * 10 ** 5) * days

	@staticmethod
	def calc_longwave(t_air, t_surf, cloudiness):
		lwu = 0.98 * 5.669 * 10 ** -8 * to_kelvin(t_surf) ** 4
		lwd = (0.765 + 0.22 * cloudiness ** 3) * 5.669 * 10 ** -8 * to_kelvin(t_air) ** 4
		return lwd - lwu

	@staticmethod
	def calc_shortwave(k, potential_incoming_solar_radiation):
		return potential_incoming_solar_radiation * k

	@staticmethod
	def _load_raster(raster_path, crop_path, remove_negatives=False, remove_outliers=False):
		ds = gdal.Open(raster_path)
		crop_ds = gdal.Warp("", ds, dstSRS="+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs", format="VRT", cutlineDSName=crop_path, outputType=gdal.GDT_Float32, xRes=10, yRes=10)
		band = crop_ds.GetRasterBand(1)
		nodata = band.GetNoDataValue()
		array = band.ReadAsArray()
		array[array == nodata] = np.nan
		if remove_negatives:
			array[array < 0] = np.nan  # makes sense for albedo which couldn't be negative
		if remove_outliers:
			array[array > 1] = np.nan
		return array

	@staticmethod
	def show_me(array):
		plt.imshow(array)
		plt.colorbar()
		plt.show()
		plt.clf()

	def interpolate_air_t(self, t_air, elev, v_gradient=None):
		if v_gradient is None:
			v_gradient = -0.006  # 6 degrees Celsius or Kelvin per 1 m
		return self.interpolate_on_dem(t_air, elev, v_gradient)

	def interpolate_pressure(self, pressure, elev, v_gradient=None):
		if v_gradient is None:
			v_gradient = -0.1145  # hPa per 1 m
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
