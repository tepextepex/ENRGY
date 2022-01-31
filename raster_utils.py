import os.path
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "/home/tepex/PycharmProjects/energy/2021/raster/"


def show_me(array, title=None, units=None, show=False, dir=None, verbose=False):
    OUT_DIR = "/home/tepex/PycharmProjects/energy/2021/raster/"
    try:
        plt.imshow(array)
        mean_str = ""
        if verbose:
            mean = float(np.nanmean(array))
            print("Mean %s is %.3f:" % (title, mean))
            mean_str = " (mean = %.3f)" % mean
        if title is not None:
            plt.title("%s%s" % (title, mean_str))  # mean value plotted only if verbose mode is chosen
        cb = plt.colorbar()
        if units is not None:
            cb.set_label(units)
        if dir is not None:
            OUT_DIR = os.path.join(OUT_DIR, dir)
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
            print("Directory created: %s" % OUT_DIR)
        plt.savefig(os.path.join(OUT_DIR, "%s.png" % title))
        if show:
            plt.show()
        plt.clf()
    except Exception as e:
        print(e)


def load_raster(raster_path, crop_path, remove_negatives=False, remove_outliers=False, v=True):
    ds = gdal.Open(raster_path)
    crop_ds = gdal.Warp("", ds, dstSRS="+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs", format="VRT",
                        cutlineDSName=crop_path, cropToCutline=True, outputType=gdal.GDT_Float32, xRes=10, yRes=10)
    gt = crop_ds.GetGeoTransform()
    proj = crop_ds.GetProjection()
    band = crop_ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    array = band.ReadAsArray()
    array[array == nodata] = np.nan
    if remove_negatives:
        array[array < 0] = np.nan  # makes sense for albedo which couldn't be negative
    if remove_outliers:
        array[array > 1] = np.nan
    if v:
        print("Raster size is %dx%d" % array.shape)
    return array, gt, proj


def export_array_as_geotiff(array_to_export, geotransform, projection, path, scale_mult=None):
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
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(array)
    band.FlushCache()
    ds = None

    return path


def get_value_by_real_coords(array, gt, easting, northing):
    ul_x, x_dist, x_skew, ul_y, y_skew, y_dist = gt
    pixel = int((easting - ul_x) / x_dist)
    line = -int((ul_y - northing) / y_dist)
    return array[line][pixel]
