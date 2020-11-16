import os.path
import gdal
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "/home/tepex/PycharmProjects/energy/test_png/"


def show_me(array, title=None, units=None, show=False, verbose=False):
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
        # plt.savefig("/home/tepex/PycharmProjects/energy/png/%s.png" % title)
        plt.savefig(os.path.join(OUT_DIR, "%s.png" % title))
        if show:
            plt.show()
        plt.clf()
    except Exception as e:
        print(e)


def load_raster(raster_path, crop_path, remove_negatives=False, remove_outliers=False):
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
