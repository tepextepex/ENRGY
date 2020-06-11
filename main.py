from model import Energy
import os

arcticdem_path = "/home/tepex/AARI/Glaciology_2019/lighting/source/dem.tif"
glacier_outlines_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/shp/aldegonda_outlines_2018/aldegonda_outlines_2018.shp"
albedo_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/L8_albedo_20190727_crop.tif"  # constant albedo for the whole year
insolation_path = "/home/tepex/AARI/Glaciers/Aldegonda albedo/gridded/total_insolation_July.tif"

e = Energy(arcticdem_path, glacier_outlines_path, albedo_path, insolation_path)
e.show_me(e.base_dem_array)
# e.show_me(e.constant_albedo)
# e.show_me(e.constant_insolation)
