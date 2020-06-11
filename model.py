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
