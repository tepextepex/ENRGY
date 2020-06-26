import os
import os.path
import subprocess
from datetime import datetime


def simulate_lighting(dem_path, date, out_dir=None):
    """
    Simulates daily lighting of an every pixel for a given DEM [kW * h / m2]
    :param dem_path: path to a georeferenced raster representing Digital Elevation Model (DEM)
    :param date: datetime object or a string with a date in %Y%m%d format
    :param out_dir: the same directory where input DEM is stored, if not specified
    :return: path to a daily lighting raster if success / False if error
    """
    if isinstance(date, datetime):
        date = date.strftime("%Y%m%d")
    if out_dir is None:
        out_dir = os.path.dirname(dem_path)
    direct_path = os.path.join(out_dir, "%s_direct.sdat" % date)
    diffus_path = os.path.join(out_dir, "%s_diffus.sdat" % date)
    total_path = os.path.join(out_dir, "%s_total.sdat" % date)

    # we need to reformat date into SAGA-acceptable format:
    date = datetime.strptime(date, "%Y%m%d")
    date = date.strftime("%m/%d/%Y")

    # params = (dem_path, direct_path, diffus_path, total_path, date)
    # cmd = "saga_cmd ta_lighting 2 -GRD_DEM %s -GRD_LINKE_DEFAULT 3 -GRD_DIRECT '%s' -GRD_DIFFUS '%s' -GRD_TOTAL '%s' -SOLARCONST 1367.0 -UNITS 0 -SHADOW 1 -LOCATION 1 -PERIOD 1 -DAY %s -HOUR_STEP 0.25 -METHOD 2" % params
    params = (dem_path, total_path, date)
    cmd = "saga_cmd ta_lighting 2 -GRD_DEM %s -GRD_LINKE_DEFAULT 3 -GRD_TOTAL '%s' -SOLARCONST 1367.0 -UNITS 0 -SHADOW 1 -LOCATION 1 -PERIOD 1 -DAY %s -HOUR_STEP 0.25 -METHOD 2" % params
    # print(cmd)
    # os.system(cmd)  # DEPRECATED in Python 3, but still works
    out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()
    # parsing SAGA command-line output is too complicated
    # it's easier to check if output file was created or not:
    if os.path.isfile(total_path):
        return total_path
    else:
        return False


def cleanup_sgrd(sgrd_file_name):
    sgrd_file_set = ("mgrd", "prj", "sgrd", "sdat", "sdat.aux.xml")
    base = sgrd_file_name[:-4]
    for ext in sgrd_file_set:
        try:
            os.remove(base + ext)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    sgrd = print(simulate_lighting("/home/tepex/AARI/Glaciology_2019/lighting/source/dem.tif", "20200626"))
    # cleanup_sgrd("/home/tepex/AARI/Glaciology_2019/lighting/source/20200626_total.sgrd")
