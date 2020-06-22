import gdal
import numpy as np
import matplotlib.pyplot as plt
from turbo import calc_turbulent_fluxes
from turbo import _calc_e_max


class Energy:
	def __init__(self, base_dem_path, glacier_outlines_path, albedo_path, potential_incoming_radiation_path):

		self._init_constants()

		self.outlines_path = glacier_outlines_path

		print("Loading base DEM...")
		self.base_dem_array = self._load_raster(base_dem_path, self.outlines_path)

		print("Loading albedo map...")
		self.albedo = self._load_raster(albedo_path, self.outlines_path, remove_outliers=True)
		show_me(self.albedo, "Albedo grid")

		print("Loading insolation map...")
		self.potential_incoming_radiation = self._load_raster(potential_incoming_radiation_path, self.outlines_path)
		show_me(self.potential_incoming_radiation, "Total potential incoming radiation", units="kW*h month-1")

	def _init_constants(self):
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

		self.t_surf = 0  # we assume that surface of melting ice is 0 degree Celsius
		self.distribute_aws_to_surface()

	def run(self, days):
		Qm = self.calc_heat_influx()  # heat available for melt:
		show_me(Qm, title="Heat available for melt", units="W m-2")

		ice_melt = self.calc_ice_melt(Qm, days)
		show_me(ice_melt, title="Ice melt", units="m w.e.")

		return ice_melt

	def calc_heat_influx(self):
		"""
		Computes an amount of heat available for melt
		:return: heat flux [W m-2]
		"""
		# TURBULENT HEAT FLUXES
		# at the AWS (needed to know Monin-Obukhov length L):
		sensible_flux, latent_flux, monin_obukhov_length = calc_turbulent_fluxes(self.z, self.wind_speed, self.Tz, self.P, self.rel_humidity)

		# at the whole glacier surface:
		sensible_flux_array, latent_flux_array, monin_obukhov_length = calc_turbulent_fluxes(self.z, self.wind_speed_array, self.t_air_array + 273.15, self.air_pressure_array * 100, self.rel_humidity_array, L=monin_obukhov_length)

		show_me(sensible_flux_array, title="Sensible heat flux", units="W m-2")
		self._export_array_as_geotiff(sensible_flux_array, "/home/tepex/PycharmProjects/energy/gtiff/sensible.tiff")

		show_me(latent_flux_array, title="Latent heat flux", units="W m-2")
		self._export_array_as_geotiff(latent_flux_array, "/home/tepex/PycharmProjects/energy/gtiff/latent.tiff")

		# LONGWAVE RADIATION FLUX
		rl = self.calc_longwave()
		show_me(rl, title="Longwave", units="W m-2")

		# SHORTWAVE RADIATION FLUX
		rs = self.calc_shortwave()
		show_me(rs, title="Incoming shortwave * (1 - albedo)", units="W m-2")

		return rl + rs + sensible_flux_array + latent_flux_array

	def calc_shortwave(self):
		self.incoming_shortwave = self.J_to_W(self.kWh_to_J(self.potential_incoming_radiation))
		# potential incoming solar radiation into real:
		self.incoming_shortwave = self.incoming_shortwave * 0.5 / 30  # 30 days
		show_me(self.incoming_shortwave, title="Real incoming solar radiation", units="W m-2")
		return self.incoming_shortwave * (1 - self.albedo)

	def calc_ice_melt(self, ice_heat_influx, days):
		"""
		Computes a melt ice layer [m w.e.]
		:param ice_heat_influx:
		:param days: number of days
		:return: thickness of melt ice layer in meters w.e.
		"""
		latent_heat_of_fusion = self.CONST["latent_heat_of_fusion"]
		ice_density = self.CONST["ice_density"]
		ice_melt = (ice_heat_influx * 86400) / (ice_density * latent_heat_of_fusion) * days

		if type(ice_melt) == np.ndarray:
			ice_melt[ice_melt < 0] = 0  # since negative ice melt (i.e. ice accumulation) is not possible
		else:
			ice_melt = 0 if ice_melt < 0 else ice_melt

		return ice_melt

	def calc_longwave(self):
		lwu = 0.98 * 5.669 * 10 ** -8 * to_kelvin(self.t_surf) ** 4
		lwd = (0.765 + 0.22 * self.cloudiness ** 3) * 5.669 * 10 ** -8 * to_kelvin(self.t_air_array) ** 4
		return lwd - lwu

	def _load_raster(self, raster_path, crop_path, remove_negatives=False, remove_outliers=False):
		ds = gdal.Open(raster_path)
		crop_ds = gdal.Warp("", ds, dstSRS="+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs", format="VRT", cutlineDSName=crop_path, cropToCutline=True, outputType=gdal.GDT_Float32, xRes=10, yRes=10)
		self.geotransform = crop_ds.GetGeoTransform()
		self.projection = crop_ds.GetProjection()
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

	def _export_array_as_geotiff(self, array_to_export, path, scale_mult=None):
		array = np.copy(array_to_export)  # to avoid modification of original array

		if scale_mult is not None:
			array = array * scale_mult
			array = np.rint(array)
			nodata = -32768
			gdt_type = gdal.GDT_Int16
		else:
			nodata = -9999
			gdt_type = gdal.GDT_Float32

		array[np.isnan(array)] = nodata

		driver = gdal.GetDriverByName("GTiff")

		ds = driver.Create(path, array.shape[1], array.shape[0], 1, gdt_type)
		ds.SetGeoTransform(self.geotransform)
		ds.SetProjection(self.projection)

		band = ds.GetRasterBand(1)
		band.SetNoDataValue(nodata)
		band.WriteArray(array)
		band.FlushCache()
		ds = None

		return path

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

	def distribute_aws_to_surface(self):
		"""
		Takes meteorological values measured at the AWS and interpolates them over
		the whole glacier surface based on vertical gradients and DEM
		:return:
		"""
		self.t_air_array = self.interpolate_air_t(self.t_air_aws, self.elev_aws)
		show_me(self.t_air_array, title="Air temperature", units="degree Celsius")

		self.wind_speed_array = self.fill_array_with_one_value(self.wind_speed)  # wind speed is non-distributed and assumed constant across the surface
		show_me(self.wind_speed_array, title="Wind speed", units="m s-1")

		self.air_pressure_array = self.interpolate_pressure(self.air_pressure, self.elev_aws)
		show_me(self.air_pressure_array, title="Air pressure", units="gPa")

		e_aws = self.rel_humidity * _calc_e_max(self.t_air_aws + 273.15, self.air_pressure * 100)  # this is partial pressure of water vapour at the AWS
		e_array = self.interpolate_e(e_aws, self.elev_aws)
		show_me(e_array, title="Partial pressure of water vapour", units="Pa")

		e_max_array = _calc_e_max(self.t_air_array + 273.15, self.air_pressure_array * 100)
		show_me(e_max_array, title="Partial pressure of saturated air (e_max)", units="Pa")

		self.rel_humidity_array = e_array / e_max_array
		show_me(self.rel_humidity_array, title="Relative humidity")

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


def show_me(array, title=None, units=None, show=False, verbose=False):
	try:
		plt.imshow(array)
		if verbose:
			print("Mean %s is %.3f:" % (title, np.nanmean(array)))
		if title is not None:
			plt.title(title)
		cb = plt.colorbar()
		if units is not None:
			cb.set_label(units)
		plt.savefig("/home/tepex/PycharmProjects/energy/png/%s.png" % title)
		if show:
			plt.show()
		plt.clf()
	except Exception as e:
		print(e)
