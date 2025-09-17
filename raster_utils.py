import os.path
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

from timeit import timeit


def show_me(array, out_dir, title=None, units=None, show=False, dir=None, verbose=False):
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
            out_dir = os.path.join(out_dir, dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print("Directory created: %s" % out_dir)
        plt.savefig(os.path.join(out_dir, "%s.png" % title))
        if show:
            plt.show()
        plt.clf()
    except Exception as e:
        print(e)


# @timeit
def load_raster(raster_path, crop_path, res, remove_negatives=False, remove_outliers=False, v=True):
    ds = gdal.Open(raster_path)
    crop_ds = gdal.Warp("", ds, dstSRS="+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs", format="VRT",
                        cutlineDSName=crop_path, cropToCutline=True, outputType=gdal.GDT_Float32, xRes=res, yRes=res)
    gt = crop_ds.GetGeoTransform()
    proj = crop_ds.GetProjection()
    band = crop_ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    array = band.ReadAsArray()
    array[array == nodata] = np.nan
    if remove_negatives:
        array[array < 0] = np.nan  # makes sense for albedo which couldn't be negative
    if remove_outliers:
        array[array < 0] = 0.001  # makes sense for albedo which couldn't be negative
        array[array > 1] = 1
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
