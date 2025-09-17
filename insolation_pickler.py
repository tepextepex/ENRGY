import os.path
from glob import glob
from timeit_my import timeit
from osgeo import gdal
from raster_utils import load_raster
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# @timeit
def pickle_insolation(raster_path, outline_path, pickle_dir, res):
    array, gt, proj = load_raster(raster_path, outline_path, res, v=False)
    out_dir = os.path.join(pickle_dir, str(res))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f"{out_dir} directory created")
    try:
        # print(f"Processing {raster_path}...")
        out_path = os.path.join(out_dir, os.path.basename(raster_path))
        np.save(out_path, array)
        # print(f"{out_path} pickled")
    except Exception as e:
        print(f"Failed to export pickle into {out_path}: {repr(e)}")


# @timeit
def load_pickle(orig_file_name, pickle_dir, res):
    pkl_dir = os.path.join(pickle_dir, str(res))
    pkl_base_name = f"{os.path.basename(orig_file_name)}.npy"
    print(f"Trying to load {os.path.join(pkl_dir, pkl_base_name)}")
    arr = np.load(os.path.join(pkl_dir, pkl_base_name))
    return arr


@timeit
def pickle_all(insolation_dir, pickle_dir, outline_path, res):
    files = glob(f"{insolation_dir}/2022*.sdat")
    for f in tqdm(sorted(files), desc=f"Exporting pickles (res={res}m)", colour="green"):
        pickle_insolation(f, outline_path, pickle_dir, res)
        # arr = load_pickle(f, pickle_dir, 30)


in_dir = "/home/tepex/PycharmProjects/energy/2022/source/dem"
outline_path = "/home/tepex/AARI/Glaciers/Aldegonda_tc/shp/glaciated_area_2022.shp"
pickle_dir = "/home/tepex/PycharmProjects/energy/2022/source/pickle"


if __name__ == "__main__":
    # for res in (30, 50, 100):
    for res in (50, 100):
        pickle_all(in_dir, pickle_dir, outline_path, res)
